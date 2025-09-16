import numpy as np

from .primitives import Scene4D


def make_basic_Scene4D():
    """Builds simple 4D data for testing, analogous to naive_gaussian()"""

    # This is the constant for the 0th-degree Spherical Harmonic basis function (1 / (2 * sqrt(pi)))
    SH_C0 = 0.28209479177387814

    num_splats = 4

    # 3D static properties
    gau_xyz = np.array([
        [0, 0, 0],    # center
        [1, 0, 0],    # along x
        [0, 1, 0],    # along y
        [0, 0, 1],    # along z
    ], dtype=np.float32)

    gau_rot = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ], dtype=np.float32)

    gau_s = np.array([
        [0.03, 0.03, 0.03],  # small sphere
        [0.2, 0.03, 0.03],   # red      elongated along x
        [0.03, 0.2, 0.03],   # green    elongated along y
        [0.03, 0.03, 0.2]    # blue     elongated along z
    ], dtype=np.float32)

    gau_c = np.array([
        [1, 0, 1],   # magenta
        [1, 0, 0],   # red
        [0, 1, 0],   # green
        [0, 0, 1],   # blue
    ], dtype=np.float32)

    # This normalization is specific to the SH DC component
    # dc = (Final Color - 0.5) / C₀ 
    gau_c = (gau_c - 0.5) / SH_C0

    gau_a = np.ones((num_splats, 1), dtype=np.float32)

    # 4D Spacetime properties
    # Motion: 9 coefficients for a cubic polynomial in time (pos, vel, accel)
    gau_motion = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],    #       splat 0 (center): static
        [0, 0.5, 0, 0, 0, 0, 0, 0, 0],  # red   splat 1 (y-axis): moves along y with constant velocity
        [0, 0, 2.5, 0, 0, 0, 0, 0, 0],  # green splat 2 (z-axis): moves along z with constant velocity
        [0.5, 0, 0, 0, 0, 0, 0, 0, 0],  # blue  splat 3 (x-axis): moves along x with constant velocity
    ], dtype=np.float32)

    # Omega: 4 components for angular velocity
    gau_omega = np.array([
        [0, 0, 0, 0],       #       splat 0 (center): no rotation
        [0, 0, 2.0, 0],     # red   splat 1 (y-axis): rotates around y-axis
        [0, 0, 0, 2.0],     # green splat 2 (z-axis): rotates around z-axis
        [0, 2.0, 0, 0],     # blue  splat 3 (x-axis): rotates around x-axis
    ], dtype=np.float32)

    # TRBF center: time at which splat is most active
    # WHEN a splat appears during the t=[0.0, 1.0] timeline
    gau_trbf_center = np.array([
        [0.0], [0.25], [0.5], [0.5]
    ], dtype=np.float32)

    # TRBF scale: duration of splat's activity
    # HOW LONG a splat stays visible. A larger value means a longer fade-in/out.
    gau_trbf_scale = np.array([
        [0.5], [0.5], [0.5], [0.5]
    ], dtype=np.float32)

    return Scene4D(
        xyz=gau_xyz,
        rot=gau_rot,
        scale=gau_s,
        dc=gau_c,
        opacity=gau_a,
        motion=gau_motion,
        omega=gau_omega,
        trbf_center=gau_trbf_center,
        trbf_scale=gau_trbf_scale,
    )


def make_static_Scene4D():
    """
    A "static" 4D Gaussian which mimics the 3D implementation.
    Useful for debugging to make sure the 4D implementation is correct.
    """

    # This is the constant for the 0th-degree Spherical Harmonic basis function (1 / (2 * sqrt(pi)))
    SH_C0 = 0.28209479177387814

    num_splats = 4

    # 3D static properties
    gau_xyz = np.array([
        [0, 0, 0],    # center
        [1, 0, 0],    # along x
        [0, 1, 0],    # along y
        [0, 0, 1],    # along z
    ], dtype=np.float32)

    gau_rot = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ], dtype=np.float32)

    gau_s = np.array([
        [0.03, 0.03, 0.03],  # small sphere
        [0.2, 0.03, 0.03],   # elongated along x
        [0.03, 0.2, 0.03],   # elongated along y
        [0.03, 0.03, 0.2]    # elongated along z
    ], dtype=np.float32)

    gau_c = np.array([
        [1, 0, 1],   # magenta
        [1, 0, 0],   # red
        [0, 1, 0],   # green
        [0, 0, 1],   # blue
    ], dtype=np.float32)

    # This normalization is specific to the SH DC component
    # dc = (Final Color - 0.5) / C₀ 
    gau_c = (gau_c - 0.5) / SH_C0

    gau_a = np.ones((num_splats, 1), dtype=np.float32)

    # 4D Spacetime properties
    # Motion: 9 coefficients for a cubic polynomial in time (pos, vel, accel)
    gau_motion = np.array([
        # splat 0 (center): static
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        # splat 1 (x-axis): static
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        # splat 2 (y-axis): static
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        # splat 3 (z-axis): static
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Omega: 4 components for angular velocity
    gau_omega = np.array([
        # splat 0 (center): no rotation
        [0, 0, 0, 0],
        # splat 1 (x-axis): no rotation
        [0, 0, 0, 0],
        # splat 2 (y-axis): no rotation
        [0, 0, 0, 0],
        # splat 3 (z-axis): no rotation
        [0, 0, 0, 0],
    ], dtype=np.float32)

    # TRBF center: time at which splat is most active
    # WHEN a splat appears during the t=[0.0, 1.0] timeline
    gau_trbf_center = np.array([
        [1.0], [1.0], [1.0], [1.0]
    ], dtype=np.float32)

    # TRBF scale: duration of splat's activity
    # HOW LONG a splat stays visible. A larger value means a longer fade-in/out.
    gau_trbf_scale = np.array([
        [1.0], [1.0], [1.0], [1.0]
    ], dtype=np.float32)

    return Scene4D(
        xyz=gau_xyz,
        rot=gau_rot,
        scale=gau_s,
        dc=gau_c,
        opacity=gau_a,
        motion=gau_motion,
        omega=gau_omega,
        trbf_center=gau_trbf_center,
        trbf_scale=gau_trbf_scale,
    )

