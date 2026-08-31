"""Rotation utilities (scalar-first quaternion conventions, Section 2.2)."""
import numpy as np


def _quat_rotmat(q):
    """Quaternion [w,x,y,z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


__all__ = ["TorqueMPCConvex"]


quat_rotmat = _quat_rotmat  # public alias

__all__ = ["quat_rotmat", "_quat_rotmat"]
