import gunpowder as gp
import gpstardist
import urllib.request
import os
import argparse

parser = argparse.ArgumentParser("Simple Stardist Generator for CREMI data")
parser.add_argument("--directory", help="Directory to store data in.", type=str, default=".")
args = parser.parse_args()
directory = args.directory

# download some test data
url = "https://cremi.org/static/data/sample_A_20160501.hdf"
urllib.request.urlretrieve(url, os.path.join(directory, 'sample_A.hdf'))

# configure where to store results
result_file = "sample_A.n5"
ds_name = "stardists_downsampled"

# declare arrays to use
raw = gp.ArrayKey("RAW")
labels = gp.ArrayKey("LABELS")
stardists = gp.ArrayKey("STARDIST")

# prepare requests for scanning (i.e. chunks) and overall
scan_request = gp.BatchRequest()
scan_request[stardists] = gp.Roi(gp.Coordinate((0, 0, 0)), gp.Coordinate((40, 100, 100))*gp.Coordinate((40, 16, 16)))
request = gp.BatchRequest()  # empty request will loop over whole area with scanning

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
    grid=(1, 2, 2)
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

