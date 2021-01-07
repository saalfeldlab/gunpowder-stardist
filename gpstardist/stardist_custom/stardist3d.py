import numpy as np
from stardist.utils import _normalize_grid
from ..lib.stardist3d_custom import c_star_dist3d_uint64


def _cpp_star_dist3d(lbl, rays, unlabeled_id, max_dist, grid=(1, 1, 1), voxel_size=(1, 1, 1)):
    dz, dy, dx = rays.vertices.T
    grid = _normalize_grid(grid,3)
    if lbl.dtype == np.uint64:
        if unlabeled_id is None:
            unlabeled_id = 0
            out = False
        else:
            out = True
        if max_dist is None:
            max_dist = np.prod(lbl.shape)
            max = False
        else:
            max = True
        return c_star_dist3d_uint64(lbl, dz.astype(np.float32, copy=False), dy.astype(np.float32, copy=False),
                                    dx.astype(np.float32, copy=False), np.array(unlabeled_id).astype(lbl.dtype),
                                    int(len(rays)), float(max_dist), int(max), int(out), int(grid[0]), int(grid[1]),
                                    int(grid[2]), float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
    else:
        raise NotImplementedError("Star Dist computation not implemented for arrays of type {0:}".format(lbl.dtype))


def star_dist3d_custom(lbl, rays, unlabeled_id, max_dist, grid=(1, 1, 1), voxel_size=(1, 1, 1), mode='cpp'):
    """lbl assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""
    if mode == 'python':
        raise(NotImplementedError("python version of star dist with max distance is not yet implemented"))
    elif mode == 'cpp':
        return _cpp_star_dist3d(lbl, rays, unlabeled_id, max_dist, grid=grid, voxel_size=voxel_size)
    elif mode == 'opencl':
        raise(NotImplementedError("opencl version of star dist with max distance is not yet implemented"))
    else:
        raise(ValueError("Unknown mode %s" % mode))
