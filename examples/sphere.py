import argparse
import gpstardist
import gunpowder as gp
import numcodecs
import numpy as np
import os
import raster_geometry
import zarr


parser = argparse.ArgumentParser("Simple Stardist Generator for a sphere")
parser.add_argument("--directory", help="Directory to store data in.", type=str, default=".")
parser.add_argument("--max_dist", help="Maximum distance for stardist computation", type=float, default=40.)
args = parser.parse_args()
directory = args.directory
max_dist = args.max_dist

# generate a dataset with a binary sphere
sphere = raster_geometry.sphere(200, 70).astype(np.uint64) # image size: 200, radius: 70
f = zarr.open(os.path.join(directory, "sphere.n5"), mode="a")
f.create_dataset(name="sphere",
                 shape=sphere.shape,
                 compressor=numcodecs.GZip(6),
                 dtype=sphere.dtype,
                 chunks=(50,50, 50),
                 overwrite=True)
f["sphere"].attrs["offset"] = (0, 0, 0)
f["sphere"].attrs["resolution"] = (1, 1, 1)
f["sphere"][:] = sphere

# declare arrays to use
labels = gp.ArrayKey("LABELS")
stardists = gp.ArrayKey("STARDIST")

# prepare requests
scan_request = gp.BatchRequest()
scan_request[stardists] = gp.Roi((0, 0, 0), (50, 50, 50))
request = gp.BatchRequest()

source = gp.ZarrSource(
    os.path.join(directory, "sphere.n5"),
    datasets={
        labels: "sphere"
    }
)

# prepare node for 3D stardist generation with a maximum distance
stardist_gen = gpstardist.AddStarDist3D(
    labels,
    stardists,
    rays=96,
    anisotropy=(1, 1, 1),
    grid=(1, 1, 1),
    max_dist=max_dist,
)

# write result to a new dataset
writer = gp.ZarrWrite(
    output_dir=directory,
    output_filename="sphere.n5",
    dataset_names={
        stardists: "stardists_max{0:}".format(max_dist)
    },
    compression_type="gzip",
)

scan = gp.Scan(scan_request)

pipeline = source + stardist_gen + writer + scan

with gp.build(pipeline):
    pipeline.request_batch(request)