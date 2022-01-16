import pandas as pd
import csv
import numpy as np
import os
from io import StringIO
from awfread import awf_read

SUPPLIER = np.array(["Jackson", "Janvier", "Scanbur"])
ANALYZER_IDS = np.array(["RP", "JF", "CV"])
NOISE_SPLS = np.array(["100", "103", "105"])
ABR_TIMES = np.array(["baseline", "24h", "2w"])
NOISE_TYPES = np.array(["baseline", "DNT", "NNT"])


def load_abr(
    data_dir: str, exp_master_list: str, disp_filenames_only: bool = False
) -> pd.DataFrame:
    """Load ABR data from .csv files in specified directory tree.

    The .csv files are exported from Tucker-Davis Technologies .awf files.
    Several fields with experiment information based on the .csv filenames
    are added.

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
        LookupError: Throw error if subject id is not available in file name.

    Returns:
        pd.DataFrame: A DataFrame with all importeddata.
    """

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
                    tmp_id = (
                        "C"
                        + file.split("_")[1]
                        + "_"
                        + file.split("_")[2].split(" ")[0]
                    )
                else:
                    tmp_id = file.split("_")[1] + "_" + file.split("_")[2].split(" ")[0]
                df_temp.loc[:, "id"] = tmp_id

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
                    (key for key, value in exp_ref.items() if tmp_id in value), None
                )
                if experimenter == None:
                    raise LookupError(
                        f"Undefined experimenter for animal {tmp_id}. Add value to {exp_master_list}."
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
                df_temp.loc[:, "wave1_amp"] = (
                    df_temp.loc[:, "V1(nv)"] - df_temp.loc[:, "V2(nv)"]
                )
                df_temp = df_temp.copy().query("wave1_amp > 0")

                # Add sequential file number
                if len(df_temp) != 0:
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
    df = df.loc[
        :,
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
        ],
    ]

    return df


def add_threshold(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Add threshold data to dataframe with other ABR / wave 1 information.

    Args:
        df (pd.DataFrame): ABR data.
        data_dir (str): Directory of "ABRthesholds.csv" file.

    Returns:
        pd.DataFrame: Same as input dataframe but with thresholds added.
    """

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
    # TODO: use pd.merge instead of hard to read df.apply
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


def load_dp(path: str, exp_master_list: str, print_skipped: bool = False) -> list:
    """Load DP data from .awf files.

    The .awf files are created with BioSigRZ (Tucker-Davis Technologies) during
    experiments. The files are fully loaded, including power spectrum. Then,
    addditional information on primary components and distortion products are
    extracted (f1, f2, 2*f1-f2).

    Args:
        path (str): A string indicating the parent directory of all
            subdirectories from which to import DP .awf files.
        exp_master_list (str): Path to file with experiment-epxerimenter
            information.
        print_skipped (bool, optional): If True, filenames that are skipped for
            unclear noise_type or abr_time are displayed. Defaults to False.

    Raises:
        LookupError: Throw error if subject id is not available in file name.

    Returns:
        list: Each element of the list is a dictionary with data from a single
            .awf file.
    """

    with open(exp_master_list, "r") as f:
        reader = csv.reader(f)
        exp_ref = {}
        for row in reader:
            exp_ref[row[0]] = row[1:]

    # Loop over all files in specified directory tree and load all DP .awf files
    all_data = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".awf") and "DP" in file:

                full_file = os.path.join(root, file)

                # Information on skipped files when requested
                if "femaleeffect" in file.lower() or "evi_effect" in root.lower():
                    special = "female_effect"
                elif "2hour" in root.lower():
                    special = "2hour"
                else:
                    special = "none"
                temp_noise_type = NOISE_TYPES[
                    np.array([nt.lower() in file.lower() for nt in NOISE_TYPES])
                ]
                if temp_noise_type.size == 0:
                    if print_skipped:
                        print(f"{file} has no readable noise_type; skipping file.")
                    continue
                temp_abr_time = ABR_TIMES[
                    np.array([nt.lower() in file.lower() for nt in ABR_TIMES])
                ]
                if temp_abr_time.size == 0:
                    if print_skipped:
                        print(f"{file} has no readable abr_time; skipping file.")
                    continue

                # Experiment information and filename
                temp_supplier = SUPPLIER[
                    np.array([sup.lower() in root.lower() for sup in SUPPLIER])
                ]
                temp_substrain = {"Jackson": "CaJ", "Janvier": "JRj", "Scanbur": "Sca"}[
                    temp_supplier[0]
                ]
                temp_noise_spl = NOISE_SPLS[
                    np.array([ns.lower() in root.lower() for ns in NOISE_SPLS])
                ]
                file_info = {
                    "file_name": file,
                    "supplier": temp_supplier[0],
                    "substrain": temp_substrain,
                    "special": special,
                    "noise_spl": temp_noise_spl[0],
                    "noise_type": temp_noise_type[0],
                    "abr_time": temp_abr_time[0],
                }

                first_char = file.split("_")[1][0].upper()
                if (
                    first_char != "C"
                    and first_char != "E"
                    and first_char != "I"
                    and first_char != "B"
                    and first_char != "X"
                ):
                    tmp_id = (
                        "C"
                        + file.split("_")[1]
                        + "_"
                        + file.split("_")[2].split(" ")[0]
                    )
                else:
                    tmp_id = file.split("_")[1] + "_" + file.split("_")[2].split(" ")[0]
                file_info["id"] = tmp_id

                experimenter = next(
                    (key for key, value in exp_ref.items() if tmp_id in value),
                    None,
                )
                if experimenter == None:
                    raise LookupError(
                        f"Undefined experimenter for animal {tmp_id}. Add value to {exp_master_list}."
                    )
                file_info["experimenter_id"] = experimenter

                # Actual loading of data
                awfdata = awf_read(full_file)
                awfdata["file_info"] = file_info

                # Detect peaks in spectrum
                awfdata = get_peaks(awfdata)

                # Add to output argument
                all_data.append(awfdata)

    # Check for repetition of filenames
    check_filenames(all_data)

    # Convert to DataFrame
    df = dp_peaks_to_df(all_data)

    # Clean up
    df.loc[:, "file_name"] = df.loc[:, "file_name"].cat.codes
    df = df.rename(
        columns={
            "file_name": "file_number",
            "AudFreq": "freq_hz",
            "L1": "level_db",
            "F1": "f1",
            "F2": "f2",
        }
    )
    df = df.drop(labels=["supplier", "recn"], axis=1)

    # Convert to Zeitgeber Time
    df.loc[:, "noise_type"] = df.loc[:, "noise_type"].map(
        {
            "baseline": "baseline",
            "DNT": "ZT3",
            "NNT": "ZT15",
        }
    )

    return df


def get_peaks(datadict: dict) -> dict:
    """Get peaks of power spectrum.

    Given the dict-form of a TDT .awf file, several parameters characterizing
    f1, f2, and 2*f1-f2 are added to the input dictionary. These data can be
    found in, e.g.,

        datadict["groups"][i]["peaks"]

    The data is extracted from the cursors in the .awf file.

    Args:
        datadict (dict): A dictionary of a TDT .awf file read with awf_read and
            to which the "peaks" information will be added.

    Returns:
        dict: The input dictionary with 'peaks' information added.
    """

    TDT_CORRECTION = -80  # 80 dB correction for Tucker-Davis Technologies reasons

    # Loop over maximum nr. of traces
    for trace in range(0, 30):
        group = datadict["groups"][trace]

        # Only continue if record is not empty, i.e., it has a power spectrum stored in 'wave'
        if any(group["wave"]):

            # Get cursors
            primary1 = group["cursors"][0]
            primary2 = group["cursors"][1]
            dist_prod = group["cursors"][2]

            # Construct dictionary with useful values
            peaks = {
                **datadict["file_info"],
                "recn": group["recn"],
                group["var_name2"][0]: group["vars"][1],
                group["var_name3"][0]: group["vars"][2],
                group["var_name5"][0]: group["vars"][4],
                group["var_name6"][0]: group["vars"][5],
                "level_f1": primary1["val"] - TDT_CORRECTION,
                "level_f2": primary2["val"] - TDT_CORRECTION,
                "level_distprod": dist_prod["val"] - TDT_CORRECTION,
            }
        else:
            peaks = {}

        # Insert peaks into datadict
        datadict["groups"][trace]["peaks"] = peaks

    return datadict


def check_filenames(all_data: list):
    """Check loaded DP data for filename conflicts.

    Args:
        all_data (list): List of dicts with DP data read from .awf files.

    Raises:
        ValueError: Raises error if a number of unique filenames and number of
            .awf files is not the same.
    """

    n_filenames = np.unique([d["file_info"]["file_name"] for d in all_data]).size
    if n_filenames != np.size(all_data):
        fn_list = [d["file_info"]["file_name"] for d in all_data]
        [print(fn) for fn in fn_list if fn_list.count(fn) > 1]
        raise ValueError(
            "The number of unique filenames and loaded .awf files is not the same."
        )
    return


def dp_peaks_to_df(datadict: list) -> pd.DataFrame:
    """Create dataframe of DPOAE peaks data from dictionary.

    Information of "peaks" is stimulus level, frequency, and response of f1,
    f2, and 2*f1-f2.

    Args:
        datadict (list): A (list of) dictionaries of (an) imported TDT .awf
            file(s) from which to retrieve "peaks" information will be added.

    Returns:
        pd.DataFrame: All "peaks" information in dataframe form.
    """

    if type(datadict) == dict:
        datadict = [datadict]

    df_array = [
        pd.concat(
            [pd.Series(grp["peaks"]) for grp in d["groups"] if grp.get("peaks")], axis=1
        )
        for d in datadict
    ]
    df = pd.concat([pks for pks in df_array], axis=1).transpose().reset_index(drop=True)

    df = df.astype(
        {
            "file_name": "category",
            "supplier": "category",
            "substrain": "category",
            "special": "category",
            "noise_spl": "category",
            "noise_type": "category",
            "abr_time": "category",
            "id": "category",
            "experimenter_id": "category",
            "recn": int,
            "AudFreq": float,
            "L1": float,
            "F1": float,
            "F2": float,
            "level_f1": float,
            "level_f2": float,
            "level_distprod": float,
        }
    )

    return df
