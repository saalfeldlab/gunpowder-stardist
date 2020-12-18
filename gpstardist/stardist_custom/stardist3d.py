import numpy as np
from stardist.utils import _normalize_grid
from ..lib.stardist3d_custom import c_star_dist3d_max


def _cpp_star_dist3d_max(lbl, rays, max_dist, grid=(1,1,1)):
    dz, dy, dx = rays.vertices.T
    grid = _normalize_grid(grid,3)

    return c_star_dist3d_max(
        lbl.astype(np.uint16, copy=False),
        dz.astype(np.float32, copy=False),
        dy.astype(np.float32, copy=False),
        dx.astype(np.float32, copy=False),
        int(len(rays)), float(max_dist), *tuple(int(a) for a in grid)
    )


def star_dist3d_max(lbl, rays, max_dist, grid=(1,1,1), mode='cpp'):
    """lbl assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""
    if mode == 'python':
        raise(NotImplementedError("python version of star dist with max distance is not yet implemented"))
    elif mode == 'cpp':
        return _cpp_star_dist3d_max(lbl, rays, max_dist, grid=grid)
    elif mode == 'opencl':
        raise(NotImplementedError("opencl version of star dist with max distance is not yet implemented"))
    else:
        raise(ValueError("Unknown mode %s" % mode))
