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

def get_names(Npars):
    if Npars == 6:
        names = ["B", "P", "Md", "Rd", "eps", "delt", "lnprob"]
    elif Npars == 7:
        names = ["B", "P", "Md", "Rd", "eps", "delt", "f_beam", "lnprob"]
    elif Npars == 8:
        names = ["B", "P", "Md", "Rd", "eps", "delt", "dipeff", "propeff", "lnprob"]
    elif Npars == 9:
        names = ["B", "P", "Md", "Rd", "eps", "delt", "dipeff", "propeff", "f_beam", "lnprob"]
    else:
        raise ValueError("Npars must be 6, 7, 8 or 9.")

    return names

def main():
    # Parse command line args
    args = parse_args()

    # Get data files
    fdata, fstats, fout = create_filenames(args)

    # Read in MCMC info
    with open(fstats, "r") as stream:
        info = json.load(stream)

    # Get MCMC parameters and parameter names
    Npars = info["Npars"]
    Nstep = info["Nstep"]
    Nwalk = info["Nwalk"]
    names = get_names(Npars)

    # Load in data
    data = pd.read_csv(
        fdata,
        header=None,
        names=names
    )

    # Find unique values of lnprob and sort
    unique_probs = np.sort(np.unique(data["lnprob"]))[::-1]
    print(f"--> Number of unique probabilities: {len(unique_probs)}")

    # Find find first location of best Nwalk probability
    indices = []
    for prob in unique_probs[:Nwalk]:
        indices.append(np.where(data["lnprob"].values == prob)[0][0])

    # Write probabilities to a file
    print(f"--> Writing best {Nwalk} probabilities to file: {fout}")
    with open(fout, "w") as f:
        for idx in indices:
            for i, name in enumerate(names[:-1]):
                if i == (Npars - 1):
                    f.write(f"{data[name].values[idx]:.6f}\n")
                else:
                    f.write(f"{data[name].values[idx]:.6f},")

if __name__ == "__main__":
    main()
