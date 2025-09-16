import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from IPython.display import HTML

from functools import partial
import concurrent.futures
from tqdm import tqdm
import scipy as sp

from .camera import Camera
from .primitives import Splat4D

class Framebuffer:
    """Represents the output canvas for rendering."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Initialize buffers
        self.bitmap = np.zeros((height, width, 3), dtype=np.float32)
        self.alphas = np.zeros((height, width), dtype=np.float32)

    def clear(self):
        """Resets the buffers for a new frame."""
        self.bitmap.fill(0)
        self.alphas.fill(0)

    def get_image(self):
        """Returns the final color image."""
        return self.bitmap

class VertexProcessor:
    """Handles the Vertex Shader stage of the pipeline."""

    @staticmethod
    def _get_dt(splat: Splat4D, t):
        return t - splat.trbf_center

    @staticmethod
    def _get_pos_offset(splat: Splat4D, t):
        dt = VertexProcessor._get_dt(splat, t)
        c1, c2, c3 = splat.motion[0:3], splat.motion[3:6], splat.motion[6:9]
        return c1 * dt + c2 * (dt**2) + c3 * (dt**3)
    
    @staticmethod
    def _get_rot_offset(splat: Splat4D, t):
        return splat.omega * VertexProcessor._get_dt(splat, t)

    @staticmethod
    def _get_S(splat: Splat4D):
        return np.diag(splat.scale)

    @staticmethod
    def _get_R(splat: Splat4D, rot_offset):
        rot_t = splat.rot + rot_offset
        rot_norm = np.linalg.norm(rot_t)
        if rot_norm == 0:
            return np.identity(3)
        rot_t /= rot_norm
        w, x, y, z = rot_t
        return sp.spatial.transform.Rotation.from_quat([x, y, z, w]).as_matrix()

    @staticmethod
    def _get_cov3d(S, R):
        return R @ S @ S.T @ R.T

    @staticmethod
    def _get_cov2d(cov3d, view_mat, view_space, fy):
        focal_x = fy
        z_inv = 1.0 / view_space[2]
        z_inv2 = z_inv * z_inv
        J = np.array([
            [focal_x * z_inv, 0.0, -(focal_x * view_space[0]) * z_inv2],
            [0.0, fy * z_inv, -(fy * view_space[1]) * z_inv2],
            [0.0, 0.0, 0.0]
        ])
        W = view_mat[:3, :3]
        T = W.T @ J
        cov2d = T.T @ cov3d @ T
        return cov2d[:2, :2]

    @staticmethod
    def _get_ellipse_axes(cov2d):
        mid = (cov2d[0, 0] + cov2d[1, 1]) / 2.0
        radius = np.hypot((cov2d[0, 0] - cov2d[1, 1]) / 2.0, cov2d[0, 1])
        lambda1, lambda2 = mid + radius, mid - radius
        if lambda2 < 0.0:
            return None, None
        eigenvector_v = np.array([cov2d[0, 1], lambda1 - cov2d[0, 0]])
        norm_v = np.linalg.norm(eigenvector_v)
        diagonal_vector = eigenvector_v / norm_v if norm_v > 0 else np.array([1.0, 0.0])
        major_axis = np.minimum(np.sqrt(2.0 * lambda1), 1024.0) * diagonal_vector
        minor_axis = np.minimum(np.sqrt(2.0 * lambda2), 1024.0) * np.array([diagonal_vector[1], -diagonal_vector[0]])
        return major_axis, minor_axis

    @staticmethod
    def _get_shader_inputs(splat: Splat4D, opacity_t, ndc_space):
        """
        Calculates the base color and opacity that will be used as inputs for the fragment shading stage.
        """
        SH_C0 = 0.28209479177387814 # Constant for the 0th-degree SH basis function
        color_rgb = np.clip(0.5 + SH_C0 * splat.dc, 0.0, 1.0)
        depth_fade = np.clip(ndc_space[2] + 1.0, 0.0, 1.0)
        effective_opacity_t = splat.opacity[0] * opacity_t
        color_center = color_rgb * depth_fade
        opacity_center = effective_opacity_t * depth_fade
        return color_center, opacity_center

    @staticmethod
    def _get_rasterizer_inputs(splat: Splat4D, view_mat, proj_mat, w, h, cov2d, pos_offset):
        """
        Calculates the 2D conic and bounding box used by the rasterization stage.
        """
        det = np.linalg.det(cov2d)
        if det == 0.0:
            return None, None, None
        det_inv = 1.0 / det
        conic = np.array([cov2d[1, 1] * det_inv, -cov2d[0, 1] * det_inv, cov2d[0, 0] * det_inv])
        bboxsize_cam = np.array([3.0 * np.sqrt(cov2d[0, 0]), 3.0 * np.sqrt(cov2d[1, 1])])
        bboxsize_ndc = (bboxsize_cam / np.array([w, h])) * 2
        vertices = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
        bboxsize_cam_verts = np.multiply(vertices, bboxsize_cam)
        position4 = np.append(splat.xyz + pos_offset, 1.0)
        g_pos_view = view_mat @ position4
        g_pos_screen = proj_mat @ g_pos_view
        g_pos_screen /= g_pos_screen[3]
        bbox_ndc_verts = np.multiply(vertices, bboxsize_ndc) + g_pos_screen[:2]
        bbox_ndc_verts = np.hstack((bbox_ndc_verts, np.zeros((vertices.shape[0], 2))))
        bbox_ndc_verts[:, 2:4] = g_pos_screen[2:4]
        return conic, bboxsize_cam_verts, bbox_ndc_verts

    @staticmethod
    def _vertex_shader_worker(g: Splat4D, t, view_mat, proj_mat, fy, w, h):
        """
        A worker function that takes a single splat object and camera/scene information
        for a specific moment in time, and prepares the splat for being drawn on the 2D screen.
        This is a static method to be used with ProcessPoolExecutor.
        """
        # --- Temporal Culling ---
        opacity_t = np.exp(-(VertexProcessor._get_dt(g, t) / g.trbf_scale)**2)
        if opacity_t < 0.02:
            return None  # The splat is temporally invisible

        # --- Calculate Time-Dependent Properties ---
        rot_offset = VertexProcessor._get_rot_offset(g, t)
        pos_offset = VertexProcessor._get_pos_offset(g, t)

        # --- Position Transformation (World -> Camera -> Clip) ---
        xyz_world_h = np.append(g.xyz + pos_offset, 1)
        view_space = view_mat @ xyz_world_h
        clip_space = proj_mat @ view_space
        depth = view_space[2]

        # --- Frustum Culling ---
        clip = 1.2 * clip_space[3]
        if (clip_space[2] < -clip or clip_space[0] < -clip or clip_space[0] > clip or
                clip_space[1] < -clip or clip_space[1] > clip):
            return None  # Splat is outside the view frustum

        # --- Covariance Calculation and Projection ---
        S = VertexProcessor._get_S(g)
        R = VertexProcessor._get_R(g, rot_offset)
        cov3d = VertexProcessor._get_cov3d(S, R)
        cov2d = VertexProcessor._get_cov2d(cov3d, view_mat, view_space, fy)
        if cov2d is None:
            return None

        # --- Find Ellipse Axes ---
        major_axis, minor_axis = VertexProcessor._get_ellipse_axes(cov2d)
        if major_axis is None:
            return None

        # --- Calculate inputs for the shader stage (final color and opacity) ---
        ndc_space = clip_space[:3] / clip_space[3]
        color_center, opacity_center = VertexProcessor._get_shader_inputs(g, opacity_t, ndc_space)

        # --- Calculate inputs for the rasterizer stage (conic, bboxsize_cam, bbox_ndc) ---
        rasterizer_inputs = VertexProcessor._get_rasterizer_inputs(g, view_mat, proj_mat, w, h, cov2d, pos_offset)
        if rasterizer_inputs is None:
            return None
        conic, bboxsize_cam, bbox_ndc = rasterizer_inputs

        # --- Assemble the final output dictionary for the primitive ---
        return {
            "depth": depth,
            "cov2d": cov2d,
            "pos_offset": pos_offset,
            "opacity_center": opacity_center,
            "color_center": color_center,
            "conic": conic,
            "bboxsize_cam": bboxsize_cam,
            "bbox_ndc": bbox_ndc, 
            "major_axis": major_axis,   # used by the _render_frame_matplotlib rendering path
            "minor_axis": minor_axis,   # used by the _render_frame_matplotlib rendering path
            "ndc_space": ndc_space      # used by the _render_frame_matplotlib rendering path
        }

    def __init__(self):
        self.executor = concurrent.futures.ProcessPoolExecutor()

    def run_vertex_stage(self, splats, camera, t):
        """
        Takes raw splat objects and processes them in parallel to produce
        a list of 2D primitives ready for rasterization.
        """
        h, w = camera.h, camera.w
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix()
        fy = camera.get_focal()
        
        worker_fn = partial(VertexProcessor._vertex_shader_worker, t=t, view_mat=view_mat, proj_mat=proj_mat, fy=fy, w=w, h=h)
        
        print(f"Number of workers being used: {self.executor._max_workers}")
        results_iterator = self.executor.map(worker_fn, splats)
        results = list(tqdm(results_iterator, total=len(splats), desc="Vertex Processing"))
        
        visible_splats = [r for r in results if r is not None]
        
        # Sort splats from back to front for correct alpha blending
        visible_splats.sort(key=lambda p: p["depth"], reverse=False)
        
        return visible_splats

    def close(self):
        """Shuts down the process pool executor."""
        print("Shutting down vertex processor...")
        self.executor.shutdown()

class FragmentProcessor:
    """Handles the Fragment Shading stage of the pipeline."""

    def shade_fragment_naive(self, x, y, x_cam, y_cam, primitive_data, framebuffer: Framebuffer):
        """
        Shades a single pixel on the framebuffer. (Loop-based version)
        This implementation more closely reflects how things would be implemented inside an actual GPU shader.
        """
        A, B, C = primitive_data["conic"]
        color = primitive_data["color_center"]
        opacity_center = primitive_data["opacity_center"]

        # Gaussian is typically calculated as f(x, y) = A * exp(-(a*x^2 + 2*b*x*y + c*y^2))
        power = -(A*x_cam**2 + C*y_cam**2)/2.0 - B * x_cam * y_cam
        if power > 0.0:
            return

        alpha = opacity_center * np.exp(power)
        alpha = min(0.99, alpha)
        if opacity_center < 1.0 / 255.0:
            return

        # Do alpha blending using "over" method
        old_alpha = framebuffer.alphas[y, x]
        new_alpha = alpha + old_alpha * (1.0 - alpha)
        if isinstance(new_alpha, np.ndarray):
            new_alpha = new_alpha[0]
        
        framebuffer.alphas[y, x] = new_alpha
        framebuffer.bitmap[y, x, :] = (color[0:3]) * alpha + framebuffer.bitmap[y, x, :] * (1.0 - alpha)

    def shade_fragments_vectorized(self, fragment_packet, framebuffer: Framebuffer):
        """
        Shades a packet of fragments using vectorized operations.
        This is an implementation optimized for speed on the CPU using NumPy.
        """
        # Unpack data from the packet
        X_pixels, Y_pixels = fragment_packet["pixel_coords"]
        X_cam, Y_cam = fragment_packet["camera_coords"]
        primitive_data = fragment_packet["primitive_data"]
        
        A, B, C = primitive_data["conic"]
        color = primitive_data["color_center"]
        opacity_center = primitive_data["opacity_center"]

        # Vectorized Gaussian computation
        power = -(A * X_cam**2 + C * Y_cam**2) / 2.0 - B * X_cam * Y_cam
        
        # Apply conditions vectorized
        valid_mask = (power <= 0.0) & (opacity_center >= 1.0 / 255.0)
        
        if not np.any(valid_mask):
            return
        
        # Compute alpha values vectorized
        alpha = opacity_center * np.exp(power)
        alpha = np.minimum(0.99, alpha)
        
        # Apply valid mask
        alpha = np.where(valid_mask, alpha, 0.0)
        
        # Vectorized alpha blending
        X_idx = X_pixels.astype(int)
        Y_idx = Y_pixels.astype(int)
        
        old_alpha = framebuffer.alphas[Y_idx, X_idx]
        old_bitmap = framebuffer.bitmap[Y_idx, X_idx, :]
        
        new_alpha = alpha + old_alpha * (1.0 - alpha)
        
        new_bitmap = color[0:3] * alpha[:, :, np.newaxis] + old_bitmap * (1.0 - alpha[:, :, np.newaxis])
        
        framebuffer.alphas[Y_idx, X_idx] = new_alpha
        framebuffer.bitmap[Y_idx, X_idx, :] = new_bitmap

class Rasterizer:
    """Handles the Rasterization stage by generating fragments for primitives."""
    def __init__(self, camera: Camera):
        self.camera = camera

    def rasterize_primitives(self, primitives):
        """
        Runs the rasterization stage, yielding a 'fragment_packet' for each visible primitive.
        This method acts as a generator.
        """
        for vertex_shader_out in tqdm(primitives, desc="Rasterizing"):
            # Use the optimized path to generate a fragment packet
            fragment_packet = self._rasterize_primitive_vectorized(vertex_shader_out)
            if fragment_packet:
                yield fragment_packet

    def _rasterize_primitive_naive(self, vertex_shader_out, fragment_processor: FragmentProcessor, framebuffer: Framebuffer):
        """
        Performs similar calculations to _rasterize_primitive_vectorized but using nested loops.
        Based on plot_opacity() method from,
        https://github.com/thomasantony/splat/blob/master/notes/00_Gaussian_Projection.ipynb


        """
        # Compute the opacity of a gaussian given the camera
        conic = vertex_shader_out["conic"]
        bboxsize_cam = vertex_shader_out["bboxsize_cam"]
        bbox_ndc = vertex_shader_out["bbox_ndc"]

        screen_height, screen_width = framebuffer.height, framebuffer.width
        # converts abstract bbox_ndc (which is in a [-1, 1] coordinate system) into concrete pixel coordinates like (120, 250)
        bbox_screen = self.camera.ndc_to_pixel(bbox_ndc, screen_width, screen_height)
        
        if np.any(np.isnan(bbox_screen)):
            return

        # on-screen bounding box coordinates 
        ul = bbox_screen[0,:2]
        ur = bbox_screen[1,:2]
        lr = bbox_screen[2,:2]
        ll = bbox_screen[3,:2]
        
        y1 = int(np.floor(ul[1]))
        y2 = int(np.ceil(ll[1]))
        
        x1 = int(np.floor(ul[0]))
        x2 = int(np.ceil(ur[0]))
        nx = x2 - x1 # width of bb
        ny = y2 - y1 # heigh of bb


        # Boundaries of the splat's projected ellipse within the camera's own 2D coordinate system.
        coordxy = bboxsize_cam
        x_cam_1 = coordxy[0][0]   # ul
        x_cam_2 = coordxy[1][0]   # ur
        y_cam_1 = coordxy[1][1]   # ur (y)
        y_cam_2 = coordxy[2][1]   # lr

        # vertex_shader_out["xyz"] is the splat's center position in 3D world space
        # camera.position is the camera's position in 3D world space
        camera_dir = vertex_shader_out["xyz"] - self.camera.position # a vector pointing from the camera towards the splat

        # This is the rasterizer. It iterates over every pixel (x, y) inside the bounding box. 
        # For each pixel, it also calculates the corresponding coordinate in the camera's 2D 
        # plane (x_cam, y_cam)
        for x, x_cam in zip(range(x1, x2), np.linspace(x_cam_1, x_cam_2, nx)):
            if x < 0 or x >= self.camera.w:
                continue
            for y, y_cam in zip(range(y1, y2), np.linspace(y_cam_1, y_cam_2, ny)):
                if y < 0 or y >= self.camera.h:
                    continue

                # Instead of doing the math here, call the fragment processor
                fragment_processor.shade_fragment_naive(x, y, x_cam, y_cam, vertex_shader_out, framebuffer)

    
    def _rasterize_primitive_vectorized(self, vertex_shader_out):
        """
        Generates a "fragment packet" for a single primitive using vectorized NumPy operations.
        The packet contains all necessary data for the FragmentProcessor to shade the pixels.
        Returns a dictionary (fragment_packet) or None if the primitive is not visible.
        """
        # Compute the opacity of a gaussian given the camera
        conic = vertex_shader_out["conic"]
        bboxsize_cam = vertex_shader_out["bboxsize_cam"]
        bbox_ndc = vertex_shader_out["bbox_ndc"]

        A, B, C = conic

        screen_height, screen_width = self.camera.h, self.camera.w
        # converts abstract bbox_ndc (which is in a [-1, 1] coordinate system) into concrete pixel coordinates like (120, 250)
        bbox_screen = self.camera.ndc_to_pixel(bbox_ndc, screen_width, screen_height)
        
        if np.any(np.isnan(bbox_screen)):
            return None

        # on-screen bounding box coordinates 
        ul = bbox_screen[0,:2]
        ur = bbox_screen[1,:2]
        lr = bbox_screen[2,:2]
        ll = bbox_screen[3,:2]
        
        y1 = int(np.floor(ul[1]))
        y2 = int(np.ceil(ll[1]))
        
        x1 = int(np.floor(ul[0]))
        x2 = int(np.ceil(ur[0]))
        nx = x2 - x1 # width of bb
        ny = y2 - y1 # height of bb

        # Early exit if bounding box is invalid or too small
        if nx <= 0 or ny <= 0:
            return None

        # Boundaries of the splat's projected ellipse within the camera's own 2D coordinate system.
        coordxy = bboxsize_cam
        x_cam_1 = coordxy[0][0]   # ul
        x_cam_2 = coordxy[1][0]   # ur
        y_cam_1 = coordxy[1][1]   # ur (y)
        y_cam_2 = coordxy[2][1]   # lr


        # Clip bounding box to image boundaries
        x1_clipped = max(0, x1)
        x2_clipped = min(self.camera.w, x2)
        y1_clipped = max(0, y1)
        y2_clipped = min(self.camera.h, y2)
        
        if x1_clipped >= x2_clipped or y1_clipped >= y2_clipped:
            return None

        # Create coordinate grids for vectorized computation
        x_pixels = np.arange(x1_clipped, x2_clipped)
        y_pixels = np.arange(y1_clipped, y2_clipped)
        
        # Create meshgrid for vectorized operations
        X_pixels, Y_pixels = np.meshgrid(x_pixels, y_pixels, indexing='ij')
        
        # Map pixel coordinates to camera coordinates
        # Linear interpolation from pixel space to camera space
        x_cam = np.linspace(x_cam_1, x_cam_2, nx)
        y_cam = np.linspace(y_cam_1, y_cam_2, ny)
        
        # Map the clipped pixel coordinates to camera coordinates
        x_cam_mapped = np.interp(x_pixels, np.arange(x1, x2), x_cam)
        y_cam_mapped = np.interp(y_pixels, np.arange(y1, y2), y_cam)
        
        # Create meshgrid for camera coordinates
        X_cam, Y_cam = np.meshgrid(x_cam_mapped, y_cam_mapped, indexing='ij')
        
        # Package the results into a dictionary and return it
        return {
            "pixel_coords": (X_pixels, Y_pixels),
            "camera_coords": (X_cam, Y_cam),
            "primitive_data": vertex_shader_out
        }

class Renderer:
    def __init__(self, gauss_objs, camera: Camera, num_frames: int, figsize=(10, 10)):
        self.gauss_objs = gauss_objs
        self.camera = camera
        self.num_frames = num_frames
        self.figsize = figsize
        self.fig = None
        self.ax = None

        # --- Pipeline Components ---
        self.vertex_processor = VertexProcessor()
        self.rasterizer = Rasterizer(self.camera)
        self.fragment_processor = FragmentProcessor()
        # The framebuffer size depends on the camera.
        self.framebuffer = Framebuffer(self.camera.w, self.camera.h)

    def render_frame(self, frame_num, method='rasterized', figsize=None):
        print(f"Rendering frame {frame_num} with {method} method...")
        
        self._prepare_plot(figsize)
        
        if method == 'vectorized':
            return self._render_frame_matplotlib(frame_num)
        elif method == 'rasterized':
            return self._render_frame(frame_num)
        else:
            raise ValueError(f"Unknown rendering method: {method}")
        
        print("Rendering complete.")
        plt.show()

    def render_animation(self, interval=30, figsize=(12, 7)):
        """
        Creates and returns an animation using rasterized rendering method.
        The `_plot_frame_rasterized` methods is called for every frame of the animation.
        """
        self._prepare_plot(figsize=figsize)

        ani = animation.FuncAnimation(
            self.fig,
            self._render_frame,
            frames=self.num_frames,
            # fargs=(), # if you need to pass extra args to the function
            interval=interval,
            blit=False # Blit must be False on most backends when using ax.clear()
        )
        
        plt.close(self.fig) # Prevent the static figure from displaying
        return HTML(ani.to_jshtml())

    def close(self):
        """
        Shuts down the pipeline components. Should be called when rendering is complete
        to release resources, especially for the multiprocessing pool.
        """
        self.vertex_processor.close()
      
    def _prepare_plot(self, figsize):
        """Prepares the figure and axes for plotting."""
        if figsize is None:
            figsize = self.figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def _finalize_plot_settings(self):
        """Applies final, common settings to the plot for a given frame."""
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def _render_frame_matplotlib(self, frame_num):
        """
        This rendering path relies on Matplotlib's vector graphics engine (patches.Ellipse)
        rather than a custom rasterizer. It draws the "perfect" mathematical ellipse,
        rather than a pixel-based approximation, which is useful for debugging.
        """
        # Clear the previous frame's content
        self.ax.clear()
        
        # Calculate the current time 't' from the frame number (from 0.0 to 1.0)
        t = frame_num / (self.num_frames - 1) if self.num_frames > 1 else 0.0

        # --- Apply plot settings ---
        self.ax.set_xlim(0, self.camera.w)
        self.ax.set_ylim(self.camera.h, 0) # Invert Y-axis for screen coordinates

        # --- Core rendering logic (adapted from your plot_splats function) ---
        # Get properties for all visible splats at the current time 't'
        visible_splats = []
        view_mat = self.camera.get_view_matrix()
        proj_mat = self.camera.get_projection_matrix()
        fy = self.camera.get_focal()
        w, h = self.camera.w, self.camera.h
        for g in tqdm(self.gauss_objs, desc="Vectorized Processing"):
            splat = VertexProcessor._vertex_shader_worker(g, t, view_mat, proj_mat, fy, w, h)
            if splat is not None:
                visible_splats.append(splat)

        # Sort splats from back to front for correct alpha blending
        visible_splats.sort(key=lambda p: p["depth"], reverse=False)

        # Draw each ellipse for the current frame
        for splat in visible_splats:
            center_ndc = splat["ndc_space"]
            major_axis = splat["major_axis"]
            minor_axis = splat["minor_axis"]
            color = splat["color_center"]
            opacity = splat["opacity_center"][0] # use the float value

            # Convert center from NDC to pixel coordinates
            center_px = self.camera.ndc_to_pixel(center_ndc)

            # Calculate ellipse parameters for matplotlib
            width = 2 * np.linalg.norm(major_axis)
            height = 2 * np.linalg.norm(minor_axis)
            angle = np.rad2deg(np.arctan2(major_axis[1], major_axis[0]))
            # The major_axis is in a Y-up camera space. Matplotlib's plot is Y-down.
            # We flip the Y component of the axis vector to get the correct angle in the Y-down plot
            angle = np.rad2deg(np.arctan2(-major_axis[1], major_axis[0]))

            # Create and add the ellipse patch
            ellipse = Ellipse(
                xy=center_px,
                width=width,
                height=height,
                angle=angle,
                facecolor=color,
                alpha=opacity
            )
            self.ax.add_patch(ellipse)

        self._finalize_plot_settings()
        
        return self.ax,

    def _render_frame(self, frame_num):
        """
        Renders a single frame by executing the full software rendering pipeline.
        """
        # Clear the previous frame's content
        self.ax.clear()
        
        # Calculate the current time 't' from the frame number (from 0.0 to 1.0)
        t = frame_num / (self.num_frames - 1) if self.num_frames > 1 else 0.0

        # --- 1. Clear the framebuffer for the new frame ---
        self.framebuffer.clear()

        # --- 2. Vertex Processing Stage ---
        print(f"Frame {frame_num}: Running vertex processing...")
        processed_primitives = self.vertex_processor.run_vertex_stage(self.gauss_objs, self.camera, t)

        # --- 3. Rasterization Stage ---
        print(f"Frame {frame_num}: Rasterizing {len(processed_primitives)} visible splats...")
        # The rasterizer yields a packet of fragments for each primitive.
        fragment_packets = self.rasterizer.rasterize_primitives(processed_primitives)

        # The Renderer calls the fragment processor for each packet
        for packet in fragment_packets:
            self.fragment_processor.shade_fragments_vectorized(packet, self.framebuffer)

        # --- 4. Display the final image from the framebuffer ---
        final_image = self.framebuffer.get_image()
        self.ax.imshow(np.clip(final_image, 0, 1))

        self._finalize_plot_settings()
        
        return self.ax,
