import pandas as pd
import csv
import numpy as np
import os
from io import StringIO


def load_abr(data_dir: str, exp_master_list: str, disp_filenames_only: bool = False) -> pd.DataFrame:
    """Load ABR data from .csv files in specified directory tree.

    The .csv files are exported from Tucker-Davis Technologies .awf files.
    Several fields with experiment information based on the .csv filenames
    are added.

    Parameters
    ----------
    data_dir : string
        
    disp_filenames_only : {'True', 'False'}, optional
         The default is False.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing all imported and preprocessed data.

    Args:
        data_dir (str): A string indicating the parent directory from which to
            import ALL .csv files from all subdirectories. Best practice is to
            have ABR .csv files only in the direcory tree.
        exp_master_list (str): Path to file with experiment-epxerimenter
            information.
        disp_filenames_only (bool, optional): If set to True, no data is
            imported and only the filenames of the data that otherwise would be
            loaded is displayed. Used for checking errors of filename
            convention. Defaults to False.

    Raises:
        LookupError: [description]

    Returns:
        pd.DataFrame: [description]
    """
    
    SUPPLIER = np.array(["Jackson", "Janvier", "Scanbur"])
    NOISE_SPLS = np.array(["100", "103", "105"])
    ABR_TIMES = np.array(["baseline", "24h", "2w"])
    NOISE_TYPES = np.array(["baseline", "DNT", "NNT"])
    ANALYZER_IDS = np.array(["RP", "JF", "CV"])

    with open(exp_master_list, "r") as f:
        reader = csv.reader(f)
        exp_ref = {}
        for row in reader:
            exp_ref[row[0]] = row[1:]
    
    # Loop over all files in specified directory tree and use only the .csv files
    file_number = 0
    first_iter = True
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv") and "ABR" in file:
                if disp_filenames_only:
                    print(file)
                    continue
                full_file = os.path.join(root, file)

                temp_noise_type = NOISE_TYPES[
                    np.array([x.lower() in file.lower() for x in NOISE_TYPES])
                ]
                if temp_noise_type.size == 0:
                    print(file + " has no readable noise_type; skipping file.")
                    continue
                temp_abr_time = ABR_TIMES[
                    np.array([x.lower() in file.lower() for x in ABR_TIMES])
                ]
                if temp_abr_time.size == 0:
                    print(file + " has no readable abr_time; skipping file.")
                    continue

                # Start of data loading
                # Make sure there are no errors in filenames! Try displaying filenames only when
                # checking for errors.

                # Both ";" and "," were used as delimiter
                df_temp = pd.read_csv(
                    StringIO("".join(l.replace(";", ",") for l in open(full_file)))
                )
                # Get rid of illegal column names
                df_temp.rename(
                    columns={"Level(dB)": "level_db", "Freq(Hz)": "freq_hz"},
                    inplace=True,
                )
                df_temp.loc[:, "noise_type"] = temp_noise_type[0]
                df_temp.loc[:, "abr_time"] = temp_abr_time[0]

                # The following two lines are left in to detect a common error.
                if "No." in df_temp.columns:
                    print(file)
                # Exception when export to .csv didn't include Freq(Hz)
                if "Phase()" in df_temp.columns:
                    print(file)

                # Get experiment ID from filename, the .awf files do not have accurate information in field 'Sub. ID'
                first_char = file.split("_")[1][0].upper()
                if (
                    first_char != "C"
                    and first_char != "E"
                    and first_char != "I"
                    and first_char != "B"
                    and first_char != "X"
                ):
                    # split[1]: cage number (with or without 'C', 'EI', 'I', 'B')
                    # split[2]: animal number, can have trailing white space
                    tmpID = (
                        "C"
                        + file.split("_")[1]
                        + "_"
                        + file.split("_")[2].split(" ")[0]
                    )
                else:
                    tmpID = file.split("_")[1] + "_" + file.split("_")[2].split(" ")[0]
                df_temp.loc[:, "id"] = tmpID

                # Add identifier fields
                df_temp.loc[:, "supplier"] = SUPPLIER[
                    np.array([x.lower() in root.lower() for x in SUPPLIER])
                ][0]
                df_temp.loc[:, "noise_spl"] = int(
                    NOISE_SPLS[np.array([x + " dB" in root for x in NOISE_SPLS])][0]
                )
                df_temp.loc[:, "analyzer_id"] = ANALYZER_IDS[
                    np.array([x in root for x in ANALYZER_IDS])
                ][0]

                if "femaleeffect" in file.lower() or "evi_effect" in root.lower():
                    special = "female_effect"
                elif "2hour" in root.lower():
                    special = "2hour"
                else:
                    special = "none"
                df_temp.loc[:, "special"] = special

                experimenter = next(
                    (key for key, value in exp_ref.items() if tmpID in value), None
                )
                if experimenter == None:
                    raise LookupError(
                        "Undefined experimenter for animal "
                        + tmpID
                        + ". Add values to exp_master_list."
                    )
                df_temp.loc[:, "experimenter_id"] = experimenter

                # Take mean for duplicate entries and remove them
                df_temp_means = df_temp.groupby(by="level_db", as_index=False).mean()
                df_temp_means = pd.merge(
                    df_temp_means,
                    df_temp,
                    how="left",
                    on=["level_db"],
                    suffixes=["", "remove"],
                )
                df_temp = df_temp_means[df_temp.columns]
                df_temp = df_temp.drop_duplicates("level_db")

                # Add wave 1 amplitude and remove entries with negative values
                df_temp.loc[:, "wave1_amp"] = df_temp.loc[:, "V1(nv)"] - df_temp.loc[:, "V2(nv)"]
                df_temp = df_temp.copy().query("wave1_amp > 0")

                # Add sequential file number
                if len(df_temp) !=  0:
                    df_temp.loc[:, "file_number"] = file_number
                    file_number += 1

                # Create new dataframe if this was the first iteration, otherwise add to existing dataframe
                if first_iter:
                    df = df_temp
                    first_iter = False
                else:
                    df = df.append(df_temp, sort=True)
    if disp_filenames_only:
        return

    # Reindex dataframe
    df = df.reset_index(drop=True)

    # Make sure 'baseline' is the same in NoiseType and ABRtime
    df.loc[df.loc[:, "abr_time"] == "baseline", "noise_type"] = "baseline"

    # Conert to strain abbreviations
    df.loc[:, "substrain"] = df.loc[:, "supplier"].map(
        {
            "Jackson": "CaJ",
            "Janvier": "JRj",
            "Scanbur": "Sca",
        }
    )
    
    # Convert to Zeitgeber Time
    df.loc[:, "noise_type"] = df.loc[:, "noise_type"].map(
        {
            "baseline": "baseline",
            "DNT": "ZT3",
            "NNT": "ZT15",
        }
    )

    df = df.astype(
        {
            "file_number": "int32",
            "supplier": "category",
            "substrain": "category",
            "id": "category",
            "special": "category",
            "noise_spl": "int64",
            "analyzer_id": "category",
            "experimenter_id": "category",
            "noise_type": "category",
            "abr_time": "category",
            "level_db": "float64",
            "freq_hz": "int64",
            "wave1_amp": "float64",
        }
    )

    if not disp_filenames_only:
        df = add_threshold(df, data_dir)
    else:
        df = pd.DataFrame()

    # Only keep useful columns
    df = df.loc[:,
        [
            "file_number",
            "substrain",
            "id",
            "special",
            "noise_spl",
            "analyzer_id",
            "experimenter_id",
            "noise_type",
            "abr_time",
            "level_db",
            "freq_hz",
            "wave1_amp",
            "threshold",
        ]
    ]

    return df


def add_threshold(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Add threshold data to dataframe with other ABR / wave 1 information."""

    # Import threshold data
    thr_values = pd.read_csv(
        os.path.join(data_dir, "ABRthresholds.csv"),
        delimiter=";",
        usecols=range(9),
        dtype={
            "supplier": "category",
            "ID": "category",
            "ABRtime": "category",
            "analyzer_ID": "category",
            "8": "float64",
            "12": "float64",
            "16": "float64",
            "24": "float64",
            "32": "float64",
        },
    )

    # Add threshold to dataframe
    #TODO: use pd.merge instead of hard to read df.apply
    df.loc[:, "threshold"] = df.apply(
        lambda row: thr_values[
            (thr_values["supplier"] == row["supplier"])
            & (thr_values["ID"] == row["id"])
            & (thr_values["ABRtime"] == row["abr_time"])
            & (thr_values["analyzer_ID"] == row["analyzer_id"])
        ][str(int(row["freq_hz"] / 1e3))].values[0],
        axis=1,
    )

    return df