import os
import wget
import numpy as np
import pandas as pd

# Dictionary containing GRB names and web links to the data
sgrbs = {"050724": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_pEgJhv.qdp",
         "051016B": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                    "/user_objects/tprods/tmp_jsplot_wBVbSe.qdp",
         "051227": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_UA03I1.qdp",
         "060614": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_QOclPB.qdp",
         "061006": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_bAtRay.qdp",
         "061210": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_q5CHr1.qdp",
         "070714B": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                    "/user_objects/tprods/tmp_jsplot_USwo81.qdp",
         "071227": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_tyeljZ.qdp",
         "080123": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_uXodE1.qdp",
         "080503": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                   "/user_objects/tprods/tmp_jsplot_WQxIzt.qdp",
         "100212A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                    "/user_objects/tprods/tmp_jsplot_H9WTbY.qdp",
         "100522A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                    "/user_objects/tprods/tmp_jsplot_V3sVMb.qdp",
         "111121A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                    "/user_objects/tprods/tmp_jsplot_WJNTn8.qdp",
         "150424A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                    "/user_objects/tprods/tmp_jsplot_NcgG6o.qdp",
         "160410A": "http://www.swift.ac.uk/scripts/viewData.php?file=http://www.swift.ac.uk"
                    "/user_objects/tprods/tmp_jsplot_y5mK1W.qdp"}


# Change to data directory
os.chdir("data")

# Make SGRBs folder and change to it
if not os.path.exists(os.path.join(os.getcwd(), "SGRBS")):
    os.mkdir("SGRBS")
os.chdir("SGRBS")

# Loop over sgrbs list and make a directory for each grb
for grb in sgrbs.keys():

    # Check if directory exists, if not then make it
    if not os.path.exists(os.path.join(os.getcwd(), grb)):
        os.mkdir(grb)

    # Change into new directory
    os.chdir(grb)

    # Download file with wget
    filename = wget.download(sgrbs[grb])

    # Load in file
    data = np.loadtxt(os.path.join(os.getcwd(), filename),
                      comments=["!", "NO", "READ"])

    # Create data frame
    df = pd.DataFrame(data={"t": data[:, 0], "tpos": data[:, 1],
                            "tneg": data[:, 2], "flux": data[:, 3],
                            "fluxpos": data[:, 4], "fluxneg": data[:, 5]})

    # Write data frame to CSV file
    df.to_csv(os.path.join(os.getcwd(), "".join([grb, ".csv"])))

    # Remove file downloaded by wget
    os.remove(os.path.join(os.getcwd(), filename))

    # Change back to SGRBS directory
    os.chdir("..")

# Change back to root directory
os.chdir("../..")
