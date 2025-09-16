### Data Flow: The Simulated Graphics Pipeline

Here is how data flows from a `.ply` file to a final rendered image, following the classic pipeline stages as implemented in `render.py`:

#### 1\. Input Stage (Loading)

1.  **Load Data**: The process starts in `io.load_ply_4d`, which reads the raw splat data into a `Scene4D` object (SoA).
2.  **Prepare Data**: The `Renderer` takes this scene and uses `utils.scene_to_objs` to convert it into a list of `Splat4D` objects (AoS), which is easier to iterate over.

#### 2\. Vertex Processing Stage (`VertexProcessor`)

This stage is responsible for figuring out where and how each splat should appear on the 2D screen. It's executed by `vertex_processor.run_vertex_stage`, which processes all splats in parallel using `concurrent.futures.ProcessPoolExecutor`.

For each splat and a given time `t`, the `_vertex_shader_worker` function performs the following:

1.  **Temporal Culling**: It first checks if the splat is "active" at time `t`. If its temporal opacity is too low, it's discarded immediately.
2.  **Calculate Dynamics**: It computes the time-dependent position offset and rotation offset using the `motion` and `omega` properties.
3.  **World to Clip Space**: It applies the `view` and `projection` matrices from the `Camera` to transform the splat's 3D world position into 4D clip-space coordinates.
4.  **Frustum Culling**: It checks if the splat's center is outside the camera's view frustum. If so, it's discarded.
5.  **Project 3D to 2D Covariance**: This is the core of Gaussian Splatting. It constructs the 3D covariance matrix (representing the 3D ellipsoid) and projects it onto the 2D screen using the Jacobian of the perspective projection. The result is a 2D covariance matrix (`cov2d`) that mathematically describes the 2D ellipse on the screen.
6.  **Output Primitives**: If the splat is visible, the worker returns a dictionary containing everything needed for the next stages: its depth, the `cov2d` matrix, final color/opacity, and a 2D bounding box.

Finally, `run_vertex_stage` sorts all the visible primitives from back to front based on their depth. This is crucial for correct alpha blending of transparent objects.

#### 3\. Rasterization Stage (`Rasterizer`)

This stage determines which pixels on the screen are covered by each primitive.

1.  **Generate Packets**: The `rasterizer.rasterize_primitives` method iterates through the sorted primitives. 
2.  **Create Grids**: It then creates two grids: one for the pixel coordinates (`X_pixels`, `Y_pixels`) and another for the corresponding camera-space coordinates (`X_cam`, `Y_cam`) for every pixel inside the box.
3.  **Yield Fragments**: This information is bundled into a "fragment packet" and `yield`ed to the next stage.

#### 4\. Fragment Processing Stage (`FragmentProcessor`)

This stage calculates the final color of each pixel.

1.  **Shade Fragments**: The `Renderer` receives the fragment packet and passes it to `fragment_processor.shade_fragments_vectorized`.
2.  **Vectorized Calculation**: This function operates on the entire grid of pixels from the packet at once. It calculates the Gaussian falloff for every pixel based on its distance from the ellipse's center.
3.  **Alpha Blending**: It then performs the "over" alpha blending operation (`new_color = splat_color * alpha + old_color * (1 - alpha)`) to correctly composite the semi-transparent splat onto the existing colors in the framebuffer.
4.  **Update Framebuffer**: The final color and alpha values are written into the `framebuffer.bitmap` and `framebuffer.alphas` arrays.

#### 5\. Output/Display

After all primitives are processed, the `framebuffer.bitmap` holds the complete, rendered image for that frame, which is then displayed using `matplotlib.pyplot.imshow`.