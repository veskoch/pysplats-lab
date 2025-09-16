import random
import numpy as np
from .primitives import Splat4D

from .camera import Camera

def scene_to_objs(gauss_scene):
    """
    Converts a Scene4D object (Structure of Arrays) to a list of Splat4D objects (Array of Structures).

    This function iterates through the data stored in the Scene4D object, extracting the
    properties for each Gaussian splat (xyz, rotation, scale, color, opacity, motion, omega,
    trbf_center, and trbf_scale). It then creates a Splat4D object for each splat,
    populating it with the extracted properties, and appends it to a list.

    Args:
        gauss_scene (Scene4D): The Scene4D object containing the scene data in a
                                 structure-of-arrays format.

    Returns:
        list: A list of Splat4D objects, where each object represents a single
              Gaussian splat with its associated properties.  The splats are printed to console.
    """

    gauss_objs = []

    for (xyz, rot, scale, dc, opacity, motion, omega, trbf_center, trbf_scale) in \
                                    zip(gauss_scene.xyz, 
                                        gauss_scene.rot, 
                                        gauss_scene.scale, 
                                        gauss_scene.dc, 
                                        gauss_scene.opacity,
                                        gauss_scene.motion,
                                        gauss_scene.omega,
                                        gauss_scene.trbf_center,
                                        gauss_scene.trbf_scale):
        splat = Splat4D(xyz, rot, scale, dc, opacity, motion, omega, trbf_center, trbf_scale)
        gauss_objs.append(splat)
    print(f"Transformed {len(gauss_objs)} raw splats into Splat4D objects")
    return gauss_objs

def reduce_splats(splat_objects, keep_fraction=1.0):
    """
    Reduces the number of splat objects to speed up rendering, preserving order.

    Args:
        splat_objects (list): The original list of Splat4D objects.
        keep_fraction (float, optional): The fraction of splats to keep (e.g., 0.2 for 20%). 
                                         Defaults to 1.0 (keep all).

    Returns:
        list: A new, smaller list of Splat4D objects.
    """
    original_count = len(splat_objects)
    if not (0.0 <= keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be between 0.0 and 1.0")

    if keep_fraction == 1.0:
        return splat_objects
        

    num_to_keep = int(original_count * keep_fraction)
    
    # To preserve the pre-sorted order, we select random indices,
    # sort them, and then build the new list from the original.
    all_indices = list(range(original_count))
    indices_to_keep = sorted(random.sample(all_indices, k=num_to_keep))
    
    reduced_list = [splat_objects[i] for i in indices_to_keep]
        
    print(f"Reduced splats from {original_count} to {len(reduced_list)} (kept ~{keep_fraction*100:.1f}%).")
    return reduced_list


# Various utility functions used for debugging during development


def analyze_splat_locations(gaussian_objects):
    """
    Analyzes the spatial distribution of Gaussian splats.

    Args:
        gaussian_objects (list): A list of Splat4D objects.

    Returns:
        dict: A dictionary containing statistics (mean, min, max, std, center, size)
              for the xyz coordinates of the splats.
    """
    if not gaussian_objects:
        return {
            "count": 0, "mean": np.zeros(3), "min": np.zeros(3),
            "max": np.zeros(3), "std": np.zeros(3), "center": np.zeros(3),
            "size": np.zeros(3)
        }

    # Extract all xyz coordinates into a single NumPy array for efficient processing
    xyz_coords = np.array([gauss.xyz for gauss in gaussian_objects])

    # Calculate statistics
    min_pos = np.min(xyz_coords, axis=0)
    max_pos = np.max(xyz_coords, axis=0)
    
    stats = {
        "count": len(gaussian_objects),
        "mean": np.mean(xyz_coords, axis=0),
        "min": min_pos,
        "max": max_pos,
        "std": np.std(xyz_coords, axis=0),
        "center": (max_pos + min_pos) / 2.0, # Geometric center of the bounding box
        "size": max_pos - min_pos           # Dimensions of the bounding box
    }

    return stats


def create_camera_from_splat_stats(h, w, stats, view_axis='z', distance_factor=2.0):
    """
    Creates and configures a Camera to frame the scene based on splat statistics.

    Args:
        h (int): Height of the viewport.
        w (int): Width of the viewport.
        stats (dict): Statistics dictionary from analyze_splat_locations.
        view_axis (str): The axis to place the camera along ('x', 'y', or 'z').
        distance_factor (float): Multiplier for the scene size to determine camera distance.

    Returns:
        Camera: A new Camera object configured to view the splats.
    """
    camera_target = stats['center']
    
    # Use the largest dimension of the bounding box to determine camera distance
    max_scene_size = np.max(stats['size']) if np.max(stats['size']) > 0 else 1.0
    camera_distance = distance_factor * max_scene_size

    camera_pos = camera_target + np.array([0, 0, camera_distance])

    return Camera(h, w, position=camera_pos, target=camera_target)

def get_viewing_axis(camera: Camera):
    """
    Calculates the camera's viewing direction and determines the dominant axis.

    Args:
        camera (Camera): The camera object.

    Returns:
        tuple: A string for the dominant axis (e.g., "-Z") and the normalized direction vector.
    """
    # The direction is a vector pointing from the camera's position to its target
    direction = camera.target - camera.position

    # Avoid division by zero if the camera is at the target
    if np.linalg.norm(direction) == 0:
        print("Camera is at the target, no specific viewing direction.")
        return "N/A", np.array([0, 0, 0])

    # Normalize the vector to get a unit direction vector
    normalized_direction = direction / np.linalg.norm(direction)

    # Find the index of the component with the largest absolute value (0=X, 1=Y, 2=Z)
    dominant_axis_index = np.argmax(np.abs(normalized_direction))

    axes = ['X', 'Y', 'Z']
    dominant_axis = axes[dominant_axis_index]

    # Check the sign of the dominant component to determine if it's positive or negative
    if normalized_direction[dominant_axis_index] > 0:
        direction_sign = "+"
    else:
        direction_sign = "-"

    print(f"Camera is primarily looking along the {direction_sign}{dominant_axis} axis.")
    print(f"Normalized direction vector: {np.round(normalized_direction, 2)}")

    return f"{direction_sign}{dominant_axis}", normalized_direction
