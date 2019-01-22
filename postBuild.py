import os
import wget
import numpy as np
import pandas as pd

# Dictionary containing GRB names and web links to the data
sgrbs = {"050724": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_wW0vgs.qdp",
         "051016B": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_QIzTVT.qdp",
         "051227": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_kAjMHq.qdp",
         "060614": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_AgWpRH.qdp",
         "061006": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_EXLEH6.qdp",
         "061210": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_MIp216.qdp",
         "070714B": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_lAVC5A.qdp",
         "071227": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_ZlGsZ6.qdp",
         "080123": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_aewxE1.qdp",
         "080503": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_i5enQC.qdp",
         "100212A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_Jz94vr.qdp",
         "100522A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_rbWRzX.qdp",
         "111121A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_zJ5szV.qdp",
         "150424A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_utSfZc.qdp",
         "160410A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk/user_objects/tprods/tmp_jsplot_dtwcu2.qdp"}


# Change to data directory
os.chdir("data")

# Make SGRBs folder and change to it
if not os.path.exists(os.path.join(os.getcwd(), "SGRBS")):
    os.mkdir("SGRBS")
os.chdir("SGRBS")

# Loop over sgrbs list and make a directory for each grb
for grb in np.sort(sgrbs.keys()):

    # Check if directory exists, if not then make it
    if not os.path.exists(os.path.join(os.getcwd(), grb)):
        os.mkdir(grb)

    # Change into new directory
    os.chdir(grb)

    # Download file with wget
    filename = wget.download(sgrbs[grb], out="".join([grb, "_raw.txt"]))

    # Load in file
    data = np.loadtxt(filename, comments=["!", "NO", "READ"])

    # Create data frame
    df = pd.DataFrame(data={"t": data[:, 0], "tpos": data[:, 1],
                            "tneg": data[:, 2], "flux": data[:, 3],
                            "fluxpos": data[:, 4], "fluxneg": data[:, 5]})

    # Write data frame to CSV file
    df.to_csv("".join([grb, ".csv"]), index=False)

    # Remove file downloaded by wget
    os.remove(filename)

    # Change back to SGRBS directory
    os.chdir("..")

# Change back to root directory
os.chdir("../..")
