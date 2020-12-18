import logging
import stardist
import gunpowder as gp
import numpy as np

logger = logging.getLogger(__name__)
#
# class AddDistSparse(gp.BatchFilter):
#     def __init__(self, label_key, dist_key, anisotropy):
#
#     def process(self, batch, request):
#
#         lbl_data = batch.ararys[self.label_key].data
#         dist = stardist.edt_prob(lbl_data, self.anisotropy)
#
#
# class NormalizeDist(gp.BatchFilter):
#     def __init__(self, in_key, out_key):
#


class AddStarDist3D(gp.BatchFilter):
    def __init__(self, label_key, stardist_key, rays=None, mode="cpp", grid=(1, 1, 1), anisotropy=None,
                 **kwargs):
        if rays is None:
            if 'rays_json' in kwargs:
                self.rays = stardist.rays_from_json(kwargs['rays_json'])
            elif 'n_rays' in kwargs:
                self.rays = stardist.Rays_GoldenSpiral(kwargs["n_rays"])
            else:
                self.rays = stardist.Rays_GoldenSpiral(96)
        elif np.isscalar(rays):
            self.rays = stardist.Rays_GoldenSpiral(rays)
        else:
            self.rays = rays
        self.n_rays = len(self.rays)
        self.rays_json = self.rays.to_json()
        self.sd_mode = mode
        self.grid = stardist.utils._normalize_grid(grid, 3)
        self.anisotropy = anisotropy if anisotropy is None else tuple(anisotropy)
        self.ss_grid = tuple(slice(0, None, g) for g in grid)
        if 'anisotropy' in self.rays_json['kwargs']:
            if self.rays_json['kwargs']['anisotropy'] is None and self.anisotropy is not None:
                self.rays_json['kwargs']['anisotropy'] = self.anisotropy
                print("Changing 'anisotropy' of rays to %s" % str(anisotropy))
            elif self.rays_json['kwargs']['anisotropy'] != self.anisotropy:
                logger.warning("Mismatch of 'anisotropy' of rays and 'anisotropy'.")
        self.label_key = label_key
        self.stardist_key = stardist_key

    def _updated_spec(self, ref_spec):
        spec = ref_spec.copy()
        spec.dtype = np.float32
        # if stardists are on downsampled grid voxel_size needs to be adapted
        if self.grid != (1, 1, 1):
            spec.voxel_size *= gp.Coordinate(self.grid)
        return spec

    def setup(self):
        self.provides(
            self.stardist_key,
            self._updated_spec(self.spec[self.label_key])
        )

    def prepare(self, request):
        # stardist computation needs incoming labels
        deps = gp.BatchRequest()
        deps[self.label_key] = request[self.stardist_key].copy()
        return deps

    def process(self, batch, request):
        # compute stardists on label data
        data = batch.arrays[self.label_key].data
        tmp = stardist.star_dist3D(data, self.rays, mode=self.sd_mode)
        tmp = tmp[self.ss_grid]
        dist = np.moveaxis(tmp, -1, 0) # gp expects channel axis in front

        # generate spec for new batch based on what's coming in for labels
        spec = self._updated_spec(batch[self.label_key].spec)
        spec.roi = request[self.stardist_key].roi.copy()

        # assemble new array in a batch, will be added to existing batch automatically
        batch = gp.Batch()
        batch[self.stardist_key] = gp.Array(dist, spec)
        return batch
