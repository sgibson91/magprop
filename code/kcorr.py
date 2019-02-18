import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from astropy.cosmology import WMAP9 as cosmo


def k_correction(df, gamma, sigma, z, dl_cm):
    """
Function to perform a k-correction according to Bloom, Frail & Sari (2001).

    :param df: time-dependent flux data (pandas.DataFrame)
    :param gamma: photon index (float)
    :param sigma: absorption coefficient (float)
    :param z: redshift (float)
    :param dl_cm: luminosity distance - cm (float)
    :return: an array of the k-corrected data (pandas.DataFrame)
    """
    # Energy limits
    e1 = 0.3      # Low energy limit of XRT in keV
    e2 = 10.0     # High energy limit of XRT is keV
    eb = 1.0      # Bolometric bandpass lower limit
    et = 10000.0  # Bolometric bandpass upper limit

    # Calculate k coefficient
    a = ((et / (1.0 + z)) ** (2.0 - gamma)) / (2.0 - gamma)
    b = ((eb / (1.0 + z)) ** (2.0 - gamma)) / (2.0 - gamma)
    c = (e2 ** (2.0 - gamma)) / (2.0 - gamma)
    d = (e1 ** (2.0 - gamma)) / (2.0 - gamma)
    k = (a - b) / (c - d)

    # Calculate the multiplication factor
    factor = 4.0 * np.pi * (dl_cm ** 2.0) * k

    # Perform k-correction
    t_k = df["t"].values / (1.0 + z)                # Time
    tpos_k = df["tpos"].values / (1.0 + z)          # Time positive errors
    tneg_k = df["tneg"].values / (1.0 + z)          # Time negative errors
    lum = sigma * factor * df["flux"].values        # Luminosity
    lumpos = sigma * factor * df["fluxpos"].values  # Luminosity positive errors
    lumneg = sigma * factor * df["fluxneg"].values  # Luminosity negative errors

    # Convert into a dataframe
    # Important to note that output data frame will list columns in
    # alphabetical order as opposed to the order defined here
    k_data = pd.DataFrame(data={"t": t_k, "tpos": tpos_k, "tneg": tneg_k,
                                "Lum50": lum / 1.0e50,
                                "Lum50pos": lumpos / 1.0e50,
                                "Lum50neg": lumneg / 1.0e50})

    return k_data


def main(args):
    """
    :param args: command line arguments from argparse
    """
    # Read in the CSV file of parameters
    if args.type == "S":
        kcorr_df = pd.read_csv("data/kcorr_sgrbs.csv", index_col="GRB")
    elif args.type == "L":
        kcorr_df = pd.read_csv("data/kcorr_lgrbs.csv", index_col="GRB")
    else:
        print "Please provide a valid type argument.\n" \
              "The type of GRB: L - long, S - short"
        sys.exit(2)

    if args.verbose:
        print "Loading the {0}GRB properties...".format(args.type)
    grbs = kcorr_df.index.tolist()    # Get list of GRB names

    # Calculate the luminosity distance in cm and add to data frame
    kcorr_df["dl_cm"] = (cosmo.luminosity_distance(kcorr_df["z"]).value
                         * 3.08568e24)

    # Loop over GRBs
    for grb in grbs:

        # Create filepaths
        if args.type == "S":
            infile = os.path.join("data", "SGRBS", grb, "".join([grb, ".csv"]))
            outfile = os.path.join("data", "SGRBS", grb, "".join([grb,
                                   "_k.csv"]))
        elif args.type == "L":
            infile = os.path.join("data", "LGRBS", grb, "".join([grb, ".csv"]))
            outfile = os.path.join("data", "LGRBS", grb, "".join([grb,
                                                                  "_k.csv"]))
        else:
            print "Please provide a valid type argument.\n" \
                  "The type of GRB: L - long, S - short"
            sys.exit(2)

        if args.verbose:
            print "Loading data for: {0}...".format(grb)
        # Read in GRB data file
        data = pd.read_csv(infile, index_col=False)

        if args.verbose:
            print "* Performing k-correction..."
        # Perform k-correction
        k_data = k_correction(data, kcorr_df["Gamma"][grb],
                              kcorr_df["sigma"][grb], kcorr_df["z"][grb],
                              kcorr_df["dl_cm"][grb])

        # Calculate geometric mean of asymmetric errors and add to data frame
        k_data["Lum50err"] = gmean([k_data["Lum50pos"].values,
                                    np.abs(k_data["Lum50neg"].values)])

        if args.verbose:
            print "* Writing to output file..."
        # Output corrected to data
        k_data.to_csv(outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to k-correct the GRB"
                                     " samples.")
    parser.add_argument("-t", "--type", required=True,
                        help="The type of GRB: L - long, S - short")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Activates a descriptive output.")

    main(parser.parse_args())
