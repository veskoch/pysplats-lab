import numpy as np
from .primitives import Scene4D, Scene3D

from plyfile import PlyData

def load_ply_3d(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3),
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return Scene3D(xyz, rots, scales, opacities, shs)

def load_ply_4d(path) -> Scene4D:

    plydata = PlyData.read(path)
    plydata = plydata['vertex']

    print(f"{path} contains {len(plydata)} splats")

    # print(plydata)    # useful to see the properties / type of data the PLY file stores
    # print(plydata[0]) # useful to see a particular element

    # TODO Sorting
    # sorted_indices = np.argsort(
    #     -np.exp(plydata["scale_0"] + plydata["scale_1"] + plydata["scale_2"])
    #     / (1 + np.exp(-plydata["opacity"]))
    # )

    xyz = np.stack((np.asarray(plydata["x"]),
                    np.asarray(plydata["y"]),
                    np.asarray(plydata["z"])),  axis=1)
    
    trbf_center = np.asarray(plydata["trbf_center"])[..., np.newaxis]
    trbf_scale = np.asarray(plydata["trbf_scale"])[..., np.newaxis]
    opacity = np.asarray(plydata["opacity"])[..., np.newaxis]

    def _get_by_prefix(prefix):
        """Helper function to extract and stack properties from a PLY element based on a prefix."""
        prop_names = [p.name for p in plydata.properties if p.name.startswith(prefix)]
        prop_names = sorted(prop_names, key=lambda x: int(x.split('_')[-1]))    # Sort by the numeric suffix to ensure correct order (e.g., rot_0, rot_1, ...)
        if not prop_names:
            return np.empty((len(plydata.data), 0), dtype=np.float32)
        return np.stack([np.asarray(plydata[name]) for name in prop_names], axis=1)

    motion = _get_by_prefix("motion_") 
    dc = _get_by_prefix("f_dc_")
    scale = _get_by_prefix("scale_")
    rot = _get_by_prefix("rot_")
    omega = _get_by_prefix("omega_") # angular velocity

    # Activation functions
    xyz = xyz.astype(np.float32)
    rot = rot / np.linalg.norm(rot, axis=-1, keepdims=True)
    rot = rot.astype(np.float32)
    scale = np.exp(scale)
    scale = scale.astype(np.float32)
    
     # Convert f_dc from [0, 1] RGB to SH coefficients to match the rendering formula
    SH_C0 = 0.28209479177387814
    dc = (dc - 0.5) / SH_C0
    dc = dc.astype(np.float32)

    opacity = 1/(1 + np.exp(- opacity))  # sigmoid
    opacity = opacity.astype(np.float32)
    motion = motion.astype(np.float32)
    omega = omega.astype(np.float32)
    trbf_center = trbf_center.astype(np.float32)
    trbf_scale = np.exp(trbf_scale)
    trbf_scale = trbf_scale.astype(np.float32)

    # print(dc.shape)
    # print(motion[0])
    # print(trbf_scale.shape)   

    return Scene4D(xyz, rot, scale, dc, opacity, motion, omega, trbf_center, trbf_scale)

