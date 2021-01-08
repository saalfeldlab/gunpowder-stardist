import gunpowder as gp
import numpy as np
import gpstardist
import urllib.request
import os
import argparse

parser = argparse.ArgumentParser("Simple Stardist Generator for CREMI data")
parser.add_argument("--directory", help="Directory to store data in.", type=str, default=".")
parser.add_argument("--max_dist", help="Maximum distance for stardist computation", type=float, default=None)
args = parser.parse_args()
directory = args.directory
max_dist = args.max_dist

# download some test data
url = "https://cremi.org/static/data/sample_A_20160501.hdf"
urllib.request.urlretrieve(url, os.path.join(directory, 'sample_A.hdf'))

# configure where to store results
result_file = "sample_A.n5"
ds_name = "neuron_ids_stardists_downsampled"
if max_dist is not None:
    ds_name += "_max{0:}".format(max_dist)

# declare arrays to use
raw = gp.ArrayKey("RAW")
labels = gp.ArrayKey("LABELS")
stardists = gp.ArrayKey("STARDIST")

# prepare requests for scanning (i.e. chunks) and overall
scan_request = gp.BatchRequest()
scan_request[stardists] = gp.Roi(gp.Coordinate((0, 0, 0)), gp.Coordinate((40, 100, 100))*gp.Coordinate((40, 8, 8)))
voxel_size = gp.Coordinate((40, 4, 4))
request = gp.BatchRequest()  # empty request will loop over whole area with scanning
request[stardists] = gp.Roi(gp.Coordinate((40, 200, 200))*gp.Coordinate((40, 8, 8)),
                            gp.Coordinate((40, 100, 100))*gp.Coordinate((40, 8, 8))*gp.Coordinate((2, 2, 2)))
source = gp.Hdf5Source(
    os.path.join(directory, "sample_A.hdf"),
    datasets={
        labels: "volumes/labels/neuron_ids"  # reads resolution from file
    }
)

stardist_gen = gpstardist.AddStarDist3D(
    labels,
    stardists,
    rays=96,
    anisotropy=(40, 4, 4),
    grid=(1, 2, 2),
    unlabeled_id=int(np.array(-3).astype(np.uint64)),
    max_dist=max_dist,
)

writer = gp.ZarrWrite(
    output_dir=directory,
    output_filename=result_file,
    dataset_names={
        stardists: ds_name
    },
    compression_type="gzip",
)

scan = gp.Scan(scan_request)

pipeline = source + stardist_gen + writer + scan

with gp.build(pipeline):
    pipeline.request_batch(request)

