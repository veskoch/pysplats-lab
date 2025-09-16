import numpy as np
from dataclasses import dataclass


@dataclass
class Splat4D:
    """
    Represents a single Spacetime Gaussian splat as a pure data object.
    """
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    dc: np.ndarray
    opacity: np.ndarray
    motion: np.ndarray
    omega: np.ndarray
    trbf_center: np.ndarray
    trbf_scale: np.ndarray

@dataclass
class Scene3D:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        return self.sh.shape[-1]

@dataclass
class Scene4D:
    """
    Store the properties of an entire scene of splats, organized by property type.
    """

    xyz: np.ndarray # array of size (a, 3) where a is num of splats, 3 is for the xyz
    rot: np.ndarray # (a, 4)
    scale: np.ndarray # (a, 3)
    dc: np.ndarray # (a, 3)
    opacity: np.ndarray # (a, 1)
    motion: np.ndarray # (a, 9)
    omega: np.ndarray # (a, 4)
    trbf_center: np.ndarray # (a, 1)
    trbf_scale: np.ndarray # (a, 1)

    def __len__(self):
        return len(self.xyz)