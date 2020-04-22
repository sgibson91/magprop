from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
from funcs import model_lum

# Get filepaths
HERE = os.path.dirname(os.path.realpath(__file__))
tmp = HERE.split("/")

while tmp[-1] != "magprop":
    tmp.pop()

ROOT = "/".join(tmp)

GRBs = {
    "Humped": np.array([1.0, 5.0, 1.0e-3, 100.0, 0.1, 1.0]),
    "Classic": np.array([1.0, 5.0, 1.0e-3, 1000.0, 0.1, 1.0]),
    "Sloped": np.array([1.0, 1.0, 1.0e-3, 100.0, 10.0, 10.0]),
    "Stuttering": np.array([1.0, 5.0, 1.0e-5, 100.0, 0.1, 100.0]),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dataset for a synthetic GRB type"
    )

    parser.add_argument(
        "--grb",
        required=True,
        choices=["Humped", "Classic", "Sloped", "Stuttering"],
        help="GRB type name",
    )

    return parser.parse_args()


def create_filenames(GRB):
    basename = os.path.join(ROOT, "data", "synthetic_datasets")
    if not os.path.exists(basename):
        os.mkdir(basename)

    dirname = os.path.join(ROOT, basename, GRB)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    outfile = os.path.join(ROOT, dirname, "{0}.csv".format(GRB))

    return outfile


def main():
    # Parse arguments
    args = parse_args()

    # Generate filenames
    outfile = create_filenames(args.grb)

    # Get input parameters
    pars = GRBs[args.grb]

    # Generate model
    model = model_lum(pars)

    # Choose some x-y positions
    inx = np.sort(np.random.randint(low=0, high=model.shape[1], size=50))
    x = model[0, inx]
    y = model[1, inx]

    # Calculate some noise
    yerr = 0.25 * y
    y += np.random.normal(loc=0.0, scale=yerr, size=len(yerr))

    # Write out data
    data = pd.DataFrame({"x": x, "y": y, "yerr": yerr})
    data.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
