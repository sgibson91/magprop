import os
import json
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to find the best, unique probabilities"
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        choices=["L", "S"],
        help="GRB type"
    )
    parser.add_argument(
        "--grb",
        type=str,
        required=True,
        help="GRB name. Format YYMMDD[:ABC:]."
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        required=True,
        help="Label of MCMC run"
    )
    parser.add_argument(
        "--burn",
        action="store_true",
        help="Is this a burn-in run or not?"
    )

    return parser.parse_args()


def create_filenames(args):
    # Data directory
    data_dir = os.path.join("data", f"{args.type}GRBS")

    # GRB directory
    grb_dir = os.path.join(data_dir, args.grb)

    # Construct filenames
    if args.burn:
        fdata = os.path.join(grb_dir, f"{args.grb}_{args.label}_burn.csv")
    else:
        fdata = os.path.join(grb_dir, f"{args.grb}_{args.label}_chain.csv")

    fstats = os.path.join(grb_dir, f"{args.grb}_{args.label}_stats.json")
    fout = os.path.join(grb_dir, f"{args.grb}_{args.label}_bestlnp.csv")

    return fdata, fstats, fout


def main():
    # Parse command line args
    args = parse_args()

    # Get data files
    fdata, fstats, fout = create_filenames(args)

    # Read in MCMC info
    with open(fstats, "r") as stream:
        info = json.load(stream)

    # Get MCMC parameters
    Npars = info["Npars"]
    Nstep = info["Nstep"]
    Nwalk = info["Nwalk"]

    # Load in data
    data = pd.read_csv(
        fdata,
        header=None,
        names=["B", "P", "Md", "Rd", "eps", "delt", "lnprob"]
    )
    print(data.describe())


if __name__ == "__main__":
    main()
