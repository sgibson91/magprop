import os
import numpy as np
import pandas as pd

# Dictionary containing GRB names and web links to the data
sgrbs = ["050724", "051016B", "051227", "060614", "061006", "061210", "070714B",
         "071227", "080123", "080503", "100212A", "100522A", "111121A",
         "150424A", "160410A"]

# Change to data/SGRBS directory
os.chdir("data/SGRBS")

# Loop over sgrbs list and make a directory for each grb
for grb in sgrbs:

    # Check if directory exists, if not then make it
    if not os.path.exists(os.path.join(os.getcwd(), grb)):
        os.mkdir(grb)

    # Construct input and output file names
    infile = "".join([grb, "_raw.txt"])
    outfile = os.path.join(grb, "".join([grb, ".csv"]))

    # Load in file
    data = np.loadtxt(infile, comments=["!", "NO", "READ"])

    # Create data frame
    df = pd.DataFrame(data={"t": data[:, 0], "tpos": data[:, 1],
                            "tneg": data[:, 2], "flux": data[:, 3],
                            "fluxpos": data[:, 4], "fluxneg": data[:, 5]})

    # Write data frame to CSV file
    df.to_csv(outfile, index=False)

