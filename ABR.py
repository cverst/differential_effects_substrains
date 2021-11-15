# Import modules
import csv
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import StringIO
from datetime import datetime, date
import inspect
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
import seaborn as sns
import itertools

# Import R packages
rbase = rpackages.importr("base")
rgraphics = rpackages.importr("graphics")
rstats = rpackages.importr("stats")
rnlme = rpackages.importr("nlme")
rcar = rpackages.importr("car")
rlsmeans = rpackages.importr("lsmeans")


def reference_values():
    """Returns dict with hardcoded parameters per function."""

    NoiseTypes = [
        "baseline",
        "1AM",
        "3AM",
        "5AM",
        "7AM",
        "DNT",
        "11PM",
        "1PM",
        "3PM",
        "5PM",
        "7PM",
        "NNT",
        "11PM",
    ]

    # colormap
    colors = [
        (0, 0, 1),
        (1, 0.9, 0),
        (1, 0.5, 0),
        (1, 0, 0),
        (0, 0, 1),
    ]  # B -> Y -> O -> R -> B
    cmap = LinearSegmentedColormap.from_list("circadian", colors)
    colormap = dict(zip(NoiseTypes[1:], cmap(np.linspace(0, 1, len(NoiseTypes)))))
    colormap["ZT3"] = colormap.pop("DNT")
    colormap["ZT15"] = colormap.pop("NNT")
    colormap["baseline"] = np.ones(3) * 0.6
    colormap_alternate = {
        "baseline": (0.4, 0.4, 0.4),
        "ZT3": "r",
        "ZT15": "b",
        "ZT23": "r",
        "ZT11": "b",
        "ZT13": "b",
        "ZT1": "r",
    }

    # load_data_abr
    load_data_abr = {}
    #    MasterExperimentList_dir = 'G:\\Forskning\\Canlons Group\\Auditory Physiology\\Projects\\DoD\\CVanalysis\\MasterExperimentList.csv'
    MasterExperimentList_dir = "/Users/corstiaen/OneDrive - Karolinska Institutet/CVanalysis/MasterExperimentList.csv"
    with open(MasterExperimentList_dir, "r") as f:
        reader = csv.reader(f)
        experimenter = {}
        for row in reader:
            experimenter[row[0]] = row[1:]
    load_data_abr["experimenter"] = experimenter
    load_data_abr["supplier"] = np.array(["Jackson", "Janvier", "Scanbur"])
    load_data_abr["noiseSPLs"] = np.array(["100", "103", "105"])
    load_data_abr["ABRtimes"] = np.array(["baseline", "24h", "2w"])
    load_data_abr["NoiseTypes"] = np.array(NoiseTypes)
    load_data_abr["analyzerIDs"] = np.array(["RP", "JF", "CV"])

    # plot_threshold_abr
    plot_threshold_abr = {}
    plot_threshold_abr["jitter_factor"] = 1
    plot_threshold_abr["alpha"] = 0.05

    # Pool into single output
    reference_dict = {
        "colormap": colormap,
        "colormap_alternate": colormap_alternate,
        "load_data_abr": load_data_abr,
        "plot_threshold_abr": plot_threshold_abr,
    }

    return reference_dict


def load_data_abr(data_dir, confidence_interval=0.95, disp_filenames_only=False):
    """Load ABR data from .csv files in specified directory tree.

    The .csv files are exported from Tucker-Davis Technologies .asw files.
    Several fields with experiment information based on the .csv filenames
    are added. Also, outlier detection is performed, based on the I/O relation
    of wave 1 amplitude and stimulus intensity.

    Parameters
    ----------
    data_dir : string
        A string indicating the parent directory from which to import ALL .csv
        files from all subdirectories. Best practice is to have ABR .csv files
        only in the direcory tree.
    confidence_interval: scalar in range [0, 1], optional
        Confidence interval used for outlier detection. A skewed normal
        distribution is fitted to all wave 1 data within the dataset. Data points
        are identified as outlier if it lies outside the confidence interval of
        a. Default is 0.95 or 95%.
    disp_filenames_only : {'True', 'False'}, optional
        If set to True, no data is imported and only the filenames of the data
        that otherwise would be loaded is displayed. Used for error checking
        of filename convention, primarily when debugging. The default is False.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing all imported and preprocessed data.
    """

    # Starting parameters
    reference = reference_values()["load_data_abr"]
    exp_ref = reference["experimenter"]
    supplier = reference["supplier"]
    noiseSPLs = reference["noiseSPLs"]
    ABRtimes = reference["ABRtimes"]
    NoiseTypes = reference["NoiseTypes"]
    analyzerIDs = reference["analyzerIDs"]
    first_iter = True

    # Loop over all files in specified directory tree and use only the .csv files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv") and "ABR" in file:
                if disp_filenames_only:
                    print(file)
                    continue
                full_file = os.path.join(root, file)

                temp_NoiseType = NoiseTypes[
                    np.array([x.lower() in file.lower() for x in NoiseTypes])
                ]
                if temp_NoiseType.size == 0:
                    print(file + " has no readable NoiseType; skipping file.")
                    continue
                temp_ABRtime = ABRtimes[
                    np.array([x.lower() in file.lower() for x in ABRtimes])
                ]
                if temp_ABRtime.size == 0:
                    print(file + " has no readable ABRtime; skipping file.")
                    continue

                # Actual loading of data into pandas dataframe.
                # Make sure there are no errors in filenames! Try displaying filenames only when
                # checking for errors. The two if-statements do most of the checking and print a
                # message if action is needed.
                # The read_csv is a little complicated because some .csv files got ";" instead of ","
                df_temp = pd.read_csv(
                    StringIO("".join(l.replace(";", ",") for l in open(full_file)))
                )
                # Get rid of "illegal" column names
                df_temp.rename(
                    columns={"Level(dB)": "Level_dB", "Freq(Hz)": "Freq_Hz"},
                    inplace=True,
                )
                df_temp["NoiseType"] = temp_NoiseType[0]
                df_temp["ABRtime"] = temp_ABRtime[0]

                # The following two lines are left in for error checking in the datasets, but not useful
                # in regular use. I previously corrected a few files where, e.g., 'No. Avgs' was split
                # into two columns named 'No.' and 'Avgs'.
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
                    # split[1] = cage number (with or without 'C', 'EI', 'I', 'B')
                    # split[2] = animal number, can have following white space
                    tmpID = (
                        "C"
                        + file.split("_")[1]
                        + "_"
                        + file.split("_")[2].split(" ")[0]
                    )
                else:
                    tmpID = file.split("_")[1] + "_" + file.split("_")[2].split(" ")[0]
                df_temp["ID"] = tmpID

                # Add identifier fields
                df_temp["supplier"] = supplier[
                    np.array([x.lower() in root.lower() for x in supplier])
                ][0]
                df_temp["noiseSPL"] = int(
                    noiseSPLs[np.array([x + " dB" in root for x in noiseSPLs])][0]
                )
                df_temp["analyzer_ID"] = analyzerIDs[
                    np.array([x in root for x in analyzerIDs])
                ][0]

                if "femaleeffect" in file.lower() or "evi_effect" in root.lower():
                    special = "female_effect"
                elif "sham" in root.lower():
                    special = "sham"
                elif "2hour" in root.lower():
                    special = "2hour"
                else:
                    special = "none"
                df_temp["special"] = special

                experimenter = next(
                    (key for key, value in exp_ref.items() if tmpID in value), None
                )
                if experimenter == None:
                    raise LookupError(
                        "Undefined experimenter for animal "
                        + tmpID
                        + ". Add values in function experimenter_ref()."
                    )
                df_temp["experimenter_ID"] = experimenter

                # Take mean for duplicate entries and remove them
                df_temp_means = df_temp.groupby("Level_dB", as_index=False).mean()
                df_temp_means = pd.merge(
                    df_temp_means,
                    df_temp,
                    how="left",
                    on=["Level_dB"],
                    suffixes=["", "remove"],
                )
                df_temp = df_temp_means[df_temp.columns]
                df_temp = df_temp.drop_duplicates("Level_dB")

                # Add wave 1 amplitude and remove entries with negative values
                df_temp["W1amp"] = df_temp["V1(nv)"] - df_temp["V2(nv)"]
                df_temp = df_temp[df_temp["W1amp"] > 0]

                # Robust linear regression of wave 1 amplitude
                if df_temp.shape[0] > 2:
                    df_temp["RLM"] = wave1_rlm_fitting(df_temp)
                else:
                    df_temp["RLM"] = np.nan
                wave1_log = np.log(df_temp["W1amp"])
                df_temp["RLM_error"] = wave1_log - df_temp["RLM"]
                # For normalization we use unweighted std. Weighted std does not work since the average of
                # the traces is very close to zero, which in turn makes you run into numerical trouble.
                std_temp = np.std(df_temp["RLM_error"])
                df_temp["RLM_error_normalized"] = df_temp["RLM_error"] / std_temp
                df_temp["normalization_std"] = std_temp
                df_temp["W1amp_normalized"] = (wave1_log - np.mean(wave1_log)) / np.std(
                    wave1_log
                )

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

    # Outlier detection
    normalized_deviations = df["RLM_error_normalized"][
        ~np.isnan(df["RLM_error_normalized"])
    ]  # remove NaNs
    distrib_params = stats.skewnorm.fit(normalized_deviations)  # a, loc, scale
    interval = stats.skewnorm.interval(0.95, *distrib_params)
    df["is_outlier"] = np.any(
        [
            df["RLM_error_normalized"] < interval[0],
            df["RLM_error_normalized"] > interval[1],
        ],
        axis=0,
    )
    df["W1amp_no_outliers"] = df["W1amp"]
    df.loc[df["is_outlier"], "W1amp_no_outliers"] = None
    df["conf_interval_low"] = (interval[0] * df["normalization_std"]) + df["RLM"]
    df["conf_interval_high"] = (interval[1] * df["normalization_std"]) + df["RLM"]

    # Only keep useful columns
    df = df[
        [
            "supplier",
            "ID",
            "special",
            "noiseSPL",
            "analyzer_ID",
            "experimenter_ID",
            "NoiseType",
            "ABRtime",
            "Level_dB",
            "Freq_Hz",
            "W1amp",
            "W1amp_normalized",
            "RLM",
            "RLM_error",
            "RLM_error_normalized",
            "conf_interval_low",
            "conf_interval_high",
            "is_outlier",
            "W1amp_no_outliers",
        ]
    ]

    df = df.astype(
        {
            "supplier": "category",
            "ID": "category",
            "special": "category",
            "noiseSPL": "int64",
            "analyzer_ID": "category",
            "experimenter_ID": "category",
            "NoiseType": "category",
            "ABRtime": "category",
            "Level_dB": "float64",
            "Freq_Hz": "int64",
            "W1amp": "float64",
            "W1amp_normalized": "float64",
            "RLM": "float64",
            "RLM_error": "float64",
            "RLM_error_normalized": "float64",
            "conf_interval_low": "float64",
            "conf_interval_high": "float64",
            "is_outlier": "bool",
            "W1amp_no_outliers": "float64",
        }
    )

    if disp_filenames_only:
        df = []  # dummy output

    return df


def wave1_rlm_fitting(df):
    """Perform fitting of RLM to wave 1.

    Fit statsmodels.api.RLM to input data and RLM fits to DataFrame.

    In the application to ABRs outliers are detected from I/O plots, i.e.,
    plots of wave 1 amplitude vs. stimulus intensity. This relationship
    should be close to linear on a log scale. We first fit a robust linear
    regressor to the log of the wave 1 amplitude. Later we can check if a
    measurement value falls outside a confidence area from the regression
    line. If it does, it is labeled as outlier.

    Parameters
    ----------
    df : pandas.DataFrame
        Typically a DataFrame loaded with csv-export from single Tucker-Davis
        Technologies .asw file. Can be any DataFrame with fields 'Level_dB'
        and 'W1amp'.

    Returns
    -------
    fitted_values : array_like
        Estimated RLM values for the input data.
    """

    # Get data
    X = df["Level_dB"].values
    X = sm.add_constant(X)  # add intercept
    y = np.log(df["W1amp"])

    # Fit robust regression model; robust regression is insensitive to outliers
    model = sm.RLM(y, X)
    results = model.fit()

    return results.fittedvalues


def clean_data(df, data_dir):
    """Check data for known errors and clean up."""

    # Add threshold
    df = add_threshold(df, data_dir)

    # Take mean for duplicates
    df = duplicates_to_means(df)

    # Add 'is_threshold' AFTER taking means for duplicates
    df["is_threshold"] = df.apply(
        lambda row: row["Level_dB"] == row["threshold"], axis=1
    )

    # Make sure 'baseline' is the same in NoiseType and ABRtime
    df.loc[df["ABRtime"] == "baseline", "NoiseType"] = "baseline"

    # Correct wrong equipment settings
    # if df['supplier'].unique() == 'Janvier':
    # df = correct_equipment_janvier(df)
    # THIS IS NO LONGER NECESSARY. IT WAS THE WRONG GAIN SETTING THAT IS CORRECTED ELSEWHERE NOW.

    # Replace DNT and NNT with ZT3 and ZT15, respectively. And the rest...
    df.replace(
        to_replace={
            "DNT": "ZT3",
            "NNT": "ZT15",
            "5AM": "ZT23",
            "7AM": "ZT1",
            "5PM": "ZT11",
            "7PM": "ZT13",
        },
        inplace=True,
    )

    # Plot distribution as visual check on outlier detection
    plot_wave1_distribution(df)

    return df


def add_threshold(df, data_dir):
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

    # Add data to dataframe
    df["threshold"] = df.apply(
        lambda row: thr_values[
            (thr_values["supplier"] == row["supplier"])
            & (thr_values["ID"] == row["ID"])
            & (thr_values["ABRtime"] == row["ABRtime"])
            & (thr_values["analyzer_ID"] == row["analyzer_ID"])
        ][str(int(row["Freq_Hz"] / 1e3))].values[0],
        axis=1,
    )

    return df


def duplicates_to_means(df):
    """Takes the mean values for any datasets that has been analyzed more than once."""

    # Preserve datatypes by first storing them
    df_dtypes = df.dtypes
    fields_fixed = [
        "supplier",
        "ID",
        "special",
        "noiseSPL",
        "experimenter_ID",
        "NoiseType",
        "ABRtime",
        "Level_dB",
        "Freq_Hz",
    ]
    fields_mean = [
        "W1amp",
        "W1amp_normalized",
        "RLM",
        "RLM_error",
        "RLM_error_normalized",
        "conf_interval_low",
        "conf_interval_high",
        "W1amp_no_outliers",
        "threshold",
    ]

    # Merge duplicates by rather difficult to interpret way
    duplicates_merged = pd.concat(
        [
            pd.concat(
                [
                    g[fields_fixed].iloc[0],
                    np.mean(g[fields_mean]),
                    pd.Series(
                        {
                            "analyzer_ID": "|"
                            + "+".join(g["analyzer_ID"].unique())
                            + "|",
                            "is_outlier": any(g["is_outlier"]),
                        }
                    ),
                ],
                axis=0,
                sort=False,
            )
            for _, g in df.groupby(fields_fixed)
            if len(g) > 1
        ],
        axis=1,
    ).transpose()
    duplicates_merged = duplicates_merged.reindex(df.columns, axis=1)

    # Remove duplicates from df
    df = pd.concat(g for _, g in df.groupby(fields_fixed) if len(g) == 1).sort_index()

    # Add merged to df and reset index
    df = pd.concat([df, duplicates_merged], ignore_index=True)

    # Revert to original datatype
    df_dtypes["analyzer_ID"] = pd.api.types.CategoricalDtype(
        categories=df["analyzer_ID"].unique()
    )
    df = df.astype(df_dtypes)

    return df


def correct_equipment_janvier(df):
    """
    It seems the settings for ABR recordings were off in December 2018. Therefore, some
    of the Janvier data may be a factor 10 off. We find those instances below and correct
    them.
    """

    # Find wave 1 ampitudes that are off by a factor 10, so they are over 10000 uV for the
    # highest values. Group them into 'single' experiments. Because 'ID' and 'ABRtime' are
    # categorical, this yields a lot of useless NaNs. The 'count' > 0 removes the NaNs
    wrong_settings_count = (
        df[df["W1amp"] > 10000][["ID", "ABRtime", "Level_dB"]]
        .groupby(["ID", "ABRtime"])
        .agg("count")
    )
    _WRONG_EXPERIMENTS = wrong_settings_count[wrong_settings_count["Level_dB"] > 0]

    # Local function that scales wave 1 where needed
    def _scale_w1(row):
        # if _WRONG_EXPERIMENTS.index.contains(tuple(row[['ID', 'ABRtime']])):
        if tuple(row[["ID", "ABRtime"]]) in _WRONG_EXPERIMENTS.index:
            out = row["W1amp"] / 10
        else:
            out = row["W1amp"]
        return out

    def _scale_w1_no_outliers(row):
        if tuple(row[["ID", "ABRtime"]]) in _WRONG_EXPERIMENTS.index:
            out = row["W1amp_no_outliers"] / 10
        else:
            out = row["W1amp_no_outliers"]
        return out

    # Scale wave 1 amplitude
    df["W1amp"] = df.apply(_scale_w1, axis=1)
    df["W1amp_no_outliers"] = df.apply(_scale_w1_no_outliers, axis=1)

    return df


def plot_wave1_distribution(df):
    """Plot distribution of normalized wave 1 amplitude used for outlier detection."""

    n_bins = 256

    # Remove NaNs
    normalized_deviations = df["RLM_error_normalized"][
        ~np.isnan(df["RLM_error_normalized"])
    ]

    # Calculate histogram
    hist, _ = np.histogram(
        normalized_deviations, bins=n_bins, range=(-5, 5), density=True
    )

    plt.figure(figsize=[14, 4])

    # Plot the distribution
    plt.subplot(1, 2, 1)
    X = np.linspace(-5, 5, num=n_bins)
    plt.bar(X, hist, width=0.08)
    plt.xlabel("Mean and variance normalized wave 1 amplitude")
    plt.ylabel("Relative frequency")
    plt.title("Wave 1 amplitude distribution and PDF")
    # Plot PDF
    distrib_params = stats.skewnorm.fit(normalized_deviations)  # a, loc, scale
    plt.plot(X, stats.skewnorm.pdf(X, *distrib_params), "r", lw=2)
    interval = stats.skewnorm.interval(0.95, *distrib_params)
    plt.plot(interval, [0.01, 0.01], "y", lw=10)
    plt.legend(["PDF", "95% confidence interval", "Wave 1 amplitude"], loc="upper left")

    # Count number of outliers
    n_outliers = (
        df.astype({"Level_dB": "category"})
        .groupby(by=["Level_dB", "is_outlier"])
        .agg("count")["ID"]
        .fillna(0)
    )
    n_outliers_true = n_outliers.iloc[
        n_outliers.index.get_level_values("is_outlier") == True
    ]
    n_outliers_false = n_outliers.iloc[
        n_outliers.index.get_level_values("is_outlier") == False
    ]

    # Plot fraction of outliers
    plt.subplot(1, 2, 2)
    plt.bar(
        n_outliers.index.levels[0],
        n_outliers_true / (n_outliers_true.values + n_outliers_false.values),
        4,
    )
    plt.xticks(range(0, 91, 10))
    plt.xlabel("Stimulus intensity (dB SPL)")
    plt.ylabel("Fraction of outliers")
    plt.title("Fraction of outliers vs stimulus intensity")

    plt.show()

    return


def select_data(df, **kwargs):
    """Make sub-selection of DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Any pandas.Dataframe to make sub-selection from.
    **kwargs :
        Any number of 'fieldname'=value to use for selecting. value can be
        an interable. The order of arguments matters, as the first
        keyword:value pair will evaluate first, each time narrowing
        down the input dataframe further.

    Returns
    -------
    df : pandas.DataFrame
        Selection from original dataframe based on parsed values.
    """

    if kwargs is not None:
        for key, value in kwargs.items():
            if np.size(value) == 1:
                df = df[df[key] == value]
            else:
                df = df.iloc[[v in value for v in df[key]]]
    return df


def plot_io_abr(df):
    """Plot I/O functions and indicate values determined as outliers.

    Carfeul: Make sure only a single experiment is parsed."""

    # get ABRtimes and StimFreqs and order them
    uABRtimes = df["ABRtime"].unique()
    uABRtimes = np.sort([uAt[::-1] for uAt in uABRtimes])
    uABRtimes = [uAt[::-1] for uAt in uABRtimes]
    uStimFreqs = np.sort(df["Freq_Hz"].unique() / 1e3)

    # determine shape of figure
    nrows = len(uABRtimes)  # row for each ABRtime
    ncols = len(uStimFreqs)  # column for each StimFreq
    plt.figure(figsize=[3.2 * ncols, 3 * nrows])

    # Plotting commands in loop
    for count_ABRt, ABRt in enumerate(uABRtimes):
        for count_SF, SF in enumerate(uStimFreqs):
            # Narrow down dataframe
            df_temp = select_data(df, ABRtime=ABRt, Freq_Hz=SF * 1000)
            if len(df_temp) > 0:
                # Extract plot data
                X = df_temp["Level_dB"]
                Y = np.log(df_temp["W1amp"])
                mask = df_temp["is_outlier"]
                # Plotting
                plt.subplot(nrows, ncols, count_ABRt * ncols + count_SF + 1)
                plt.plot(X, Y, marker="o", linestyle="--", color="b")
                plt.plot(
                    X[mask],
                    Y[mask],
                    marker="o",
                    markeredgecolor="r",
                    markerfacecolor="r",
                    linestyle="",
                )
                plt.plot(X, df_temp["RLM"], "k:")
                plt.plot(X, df_temp["conf_interval_low"], "y:")
                plt.plot(X, df_temp["conf_interval_high"], "y:")
                # Labeling
                plt.xlabel("Stimulus intensity (dB SPL)")
                plt.title(
                    df_temp["ID"].iloc[0]
                    + ", "
                    + df_temp["NoiseType"].iloc[0]
                    + ", "
                    + str(SF)
                    + " kHz"
                )
                if count_SF == 0:
                    plt.ylabel("log(wave 1 amplitude)")

    plt.tight_layout()
    plt.show()

    return


def plot_threshold_abr(df, ylim, shift=False):
    """Create strip plot showing distribution of ABR threshold

    This function first plots different noisetypes within each frequency as separate
    strips. Subsequently, statistical tests are run to find
        1) Differences within frequency groups (Kruskal-Wallis test (nonparametric)).
        2) Differences between noisetypes within a frequency group (post hoc Mann-Whitney U).

    If input argument "shift=True" the threshold shift between different ABR times is plotted
    and tested.

    Parameters
    ----------
    df : pandas.DataFrame
        A selection of ABR data used for threshold analysis.
    """

    colormap = reference_values()["colormap_alternate"]

    reference = reference_values()["plot_threshold_abr"]
    alpha = reference["alpha"]
    jitter_factor = reference["jitter_factor"]

    uABRtimes = sorted(
        df["ABRtime"].unique()
    )  # baseline is now last, helpful for shift=True

    # Make sure all baseline are labeled as such
    if "baseline" in uABRtimes:
        df = df.reset_index(drop=True)  # needed to suppress warning
        df.loc[df["ABRtime"] == "baseline", "NoiseType"] = "baseline"

    # Only keep rows that are threshold
    colnames = [
        "supplier",
        "ID",
        "special",
        "noiseSPL",
        "analyzer_ID",
        "experimenter_ID",
        "NoiseType",
        "ABRtime",
        "Freq_Hz",
        "threshold",
    ]
    df = df[colnames].groupby(colnames[:-1]).agg(pd.Series.unique)

    # Calculate shift when queried
    if shift:
        if np.size(uABRtimes) == 1:
            raise ValueError(
                "Single ABRtime in dataframe. Increase number of ABRtimes to two or try using plot_threshold with 'shift=False'."
            )
        if np.size(uABRtimes) > 2:
            raise ValueError(
                "More than two ABRtimes in dataframe. Decrease number of ABRtimes to two."
            )
        df_reference = (
            df.reset_index(level=6)
            .xs(uABRtimes[1], level="ABRtime", drop_level=True)
            .dropna()
        )  # usually 'baseline'
        df_other = (
            df.reset_index(level=6)
            .xs(uABRtimes[0], level="ABRtime", drop_level=True)
            .dropna()
        )
        df = df_other.join(
            df_reference, how="outer", lsuffix="_other", rsuffix="_reference"
        ).dropna()
        df["threshold"] = df["threshold_other"] - df["threshold_reference"]
        df["NoiseType"] = df.apply(
            lambda row: row["NoiseType_other"] + "-" + row["NoiseType_reference"],
            axis=1,
        )

        new_colormap = {}
        for NTs in df["NoiseType"].unique():
            for NTs_other in df["NoiseType_other"].unique():
                if NTs_other in NTs:
                    new_colormap[NTs] = colormap[NTs_other]
        new_colormap["ZT13-baseline"] = "b"
        colormap = new_colormap

    df = df.dropna().reset_index()

    # Create axes
    fig = plt.figure()

    # Figure title
    u_supp = df["supplier"].unique()
    u_spec = df["special"].unique()
    u_nois = df["noiseSPL"].unique()
    u_abrt = uABRtimes
    u_anal = df["analyzer_ID"].unique()
    u_expe = df["experimenter_ID"].unique()
    if len(u_supp) > 1:
        supp_ID = "+".join(u_supp)
    else:
        supp_ID = u_supp[0]
    if len(u_spec) > 1:
        spec_ID = "+".join(u_spec)
    else:
        spec_ID = u_spec[0]
    if len(u_nois) > 1:
        nois_ID = "+".join(str(e) + " dB" for e in u_nois)
    else:
        nois_ID = u_nois[0]
    if len(uABRtimes) > 1:
        abrt_ID = "+".join(u_abrt)
    else:
        abrt_ID = u_abrt[0]
    if len(u_anal) > 1:
        anal_ID = "+".join(u_anal)
    else:
        anal_ID = u_anal[0]
    if len(u_expe) > 1:
        expe_ID = "+".join(u_expe)
    else:
        expe_ID = u_expe[0]
    fig.suptitle(
        supp_ID
        + ", "
        + spec_ID
        + ", "
        + str(nois_ID)
        + " SPL, "
        + abrt_ID
        + "\n"
        + "analyzer "
        + anal_ID
        + ", experimenter "
        + expe_ID,
        fontsize=14,
    )

    df["Freq_Hz"] /= 1e3
    df["NoiseType"] = df["NoiseType"].astype(str)
    df = df.sort_values(by=["NoiseType"], ascending=False)

    # Plotting
    df["threshold"] = np.add(
        df["threshold"],
        np.random.uniform(-jitter_factor, jitter_factor, len(df["threshold"])),
    )

    ax = plt.axes()
    df["ID"] = df["ID"].astype(str)
    df["Freq_Hz_cat"] = df["Freq_Hz"].astype("category")
    df["Freq_Hz_cat"] = df["Freq_Hz_cat"].cat.rename_categories(
        {8.0: "8", 12.0: "12", 16.0: "16", 24.0: "24", 32.0: "32"}
    )
    pale_colormap = {
        k: ([0.7, 0.7, 0.7] if k == "baseline" else v) for k, v in colormap.items()
    }
    pale_colormap = {
        k: ([1, 0.6, 0.6] if v == "r" else v) for k, v in pale_colormap.items()
    }
    pale_colormap = {
        k: ([0.6, 0.6, 1] if v == "b" else v) for k, v in pale_colormap.items()
    }

    sns.lineplot(
        ax=ax,
        x="Freq_Hz_cat",
        y="threshold",
        hue="NoiseType",
        units="ID",
        data=df,
        estimator=None,
        linewidth=0.5,
        mew=0.5,
        mfc="w",
        marker="o",
        ms=4,
        palette=pale_colormap,
        legend=False,
    )

    clrs = [plt.getp(line, "color") + [1.0] for line in ax.lines]
    for c, v in enumerate(ax.lines):
        plt.setp(v, markeredgecolor=clrs[c])
    nlineslight = len(ax.lines)

    mean_sem = (
        df[["NoiseType", "Freq_Hz", "threshold"]]
        .groupby(by=["NoiseType", "Freq_Hz"])
        .agg(["mean", "sem"])
        .reset_index()
    )
    for NT in sorted(mean_sem["NoiseType"].unique(), reverse=True):
        small_df = select_data(mean_sem, NoiseType=NT)
        ax.errorbar(
            np.arange(0, len(small_df)),
            small_df["threshold"]["mean"],
            yerr=small_df["threshold"]["sem"],
            color=colormap[NT],
            linewidth=1.5,
            capsize=4,
            marker="o",
            markersize=6,
            zorder=nlineslight + 1,
            label=NT,
        )

    plt.ylim(ylim)

    ax.legend()

    plt.xlabel("Stimulus frequency (kHz)", fontsize=14)
    if shift:
        plt.ylabel("Threshold shift (dB)", fontsize=14)
    else:
        plt.ylabel("Threshold (dB SPL)", fontsize=14)
    plt.show()

    # Statistical tests
    print("alpha = " + str(alpha) + "\n")
    for Freq_Hz in np.sort(df.Freq_Hz.unique()):
        _small_df = df.query("Freq_Hz == {f}".format(f=Freq_Hz))
        print("Frequency: " + str(Freq_Hz) + " kHz")
        kw = stats.kruskal(
            *(
                _small_df["threshold"][_small_df["NoiseType"] == NT]
                for NT in _small_df.NoiseType.unique()
            )
        )
        print("  Overall Kruskal-Wallis:")
        if kw.pvalue < alpha:
            print("   *Significant difference (p={:.4f})".format(kw.pvalue))
            print("  Post hoc Mann-Whitney U:")
            for nt_comb in itertools.combinations(_small_df.NoiseType.unique(), 2):
                mwu = stats.mannwhitneyu(
                    *(
                        _small_df["threshold"][_small_df["NoiseType"] == NT]
                        for NT in nt_comb
                    ),
                    alternative="two-sided"
                )
                if mwu.pvalue < alpha:
                    print(
                        "   *Significant effect for groups {grp0} <--> {grp1} (p={pval:.4f})".format(
                            grp0=nt_comb[0], grp1=nt_comb[1], pval=mwu.pvalue
                        )
                    )
                else:
                    print(
                        "    No significant effect for groups {grp0} <--> {grp1} (p={pval:.4f})".format(
                            grp0=nt_comb[0], grp1=nt_comb[1], pval=mwu.pvalue
                        )
                    )
        else:
            print("    No significant difference (p={:.4f})".format(kw.pvalue))

    plt.show()

    return


def fit_lme(df, weighted=False, print_results=False, qqplot=False):
    """Apply linear mixed model to ABR or DP dataframe

    Fit LME uses an R instance to compute a linear mixed effects (LME) model for the data. The
    following formula is used for ABR data:

        'log(W1amp) ~ Level_dB * NoiseType'

    This function will look at fields ID, NoiseType, Level_dB, and amp only. Any inconsistensies
    in the data, e.g., different strains or experimental conditions, are ignored.

    After fitting the LME the result is tested for heteroskedacity using Levene's test (see also
    https://en.wikipedia.org/wiki/Levene's_test). A message will be displayed if the test fails
    to show homoskedacity. Visual inspection is generally considered a better method and can be
    enabled through the input parameters. To counter heteroskedacity the data can be weighted in
    the model, again through the input parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data from which predictions will be made. It is assumed to have the following
        column names:

            ID : category
            NoiseType : category
            Level_dB : float64
            W1amp : float64

        IMPORTANT! Pandas dataframe columns MUST have appropriate datatype (i.e., categorical, float64)!

    weighted : {False, True}, optional
        Switch to enable weighting of data to counter heteroskedactiy. By default False.
    print_results : {False, True}, optional
        Switch to enable printing output of model results and statistical tests. By default False.
    qqplot : {False, True}, optional
        Switch to turn on plotting of quantile-quantile plot. Plots are stored in the working directory and
        given filenames 'tmp_*.png'. By default False.

    Returns
    -------
    predictions : pandas.DataFrame
        Dataframe with predictions of W1amp/IntensityDP for each combination of NoiseType and Level_dB.
    comparisons : pandas.DataFrame
        Dataframe with comparisons of W1amp/IntensityDP for all NoiseTypes within each Level_dB, including
        p-values of their similarity.
    model : rpy2.robject
        Instance of the model generated in R.
    """

    #    # R package names to check
    #    packnames = ('utils', 'base', 'graphics', 'stats', 'nlme', 'car', 'lsmeans')
    #    # Import R's utilities
    #    rutils = rpackages.importr('utils')
    #    # Selectively install what needs to be installed
    #    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    #    if len(names_to_install) > 0:
    #        rutils.install_packages(StrVector(names_to_install))
    # # Import R packages
    # rbase = rpackages.importr('base')
    # rgraphics = rpackages.importr('graphics')
    # rstats = rpackages.importr('stats')
    # rnlme = rpackages.importr('nlme')
    # rcar = rpackages.importr('car')
    # rlsmeans = rpackages.importr('lsmeans')

    # Convert pandas DataFrame to R dataframe
    with localconverter(default_converter + pandas2ri.converter) as cv:
        r_df = robjects.conversion.py2rpy(df)

    # The interaction term * allows for different slopes for the NoiseTypes. This syntax in R essentially expands
    # to the function 'dependent_variable = independent_variable + factor + independent_variable:factor' and allows
    # for different slopes for each NoiseType.
    # 'random=~1|ID' allows each animal ID to have a separate intercept and the intercept for a NoiseType will be
    # the mean of the intercepts of the animals within that NoiseType. The output of the model gives an estimate
    # of the variance around that estimate, which reflects the variability in the intercepts fit for each animal ID.
    formula_model = robjects.Formula("log(W1amp) ~ Level_dB * NoiseType")
    formula_weights = robjects.Formula("~1|Level_dB")
    independent_var = "Level_dB"
    formula_random = robjects.Formula("~1|ID")

    # Determine weights
    if weighted:
        weights = rnlme.varIdent(form=formula_weights)
    else:
        weights = robjects.r("NULL")

    # Fit model. REML is preferred method for smaller sample sizes.
    # ML estimates are unbiased for the fixed effects but biased for the random effects, whereas the
    # REML estimates are biased for the fixed effects and unbiased for the random effects.
    model = rnlme.lme(
        formula_model, method="REML", random=formula_random, data=r_df, weights=weights
    )
    if print_results:
        print(rstats.anova(model, type="marginal"))

    # Test for heteroskedacity
    levene_model = rcar.leveneTest(rstats.residuals(model), group=r_df.rx2("NoiseType"))
    pval_levene_model = levene_model.rx("Pr(>F)")[0][0]
    if print_results:
        print("")
        print(levene_model)
    if pval_levene_model < 0.05:
        print("====================================================================")
        print("| CAREFUL! Levene's test detects heteroskedacity in the residuals! |")
        print("====================================================================")
        print("")

    # Optional visual inspection
    if qqplot:
        plt.figure()
        stats.probplot(list(rstats.residuals(model)), dist="norm", plot=plt)
        # plt.show()
        plt.savefig(
            "tmp_"
            + str((datetime.now() - datetime(1970, 1, 1)).total_seconds())
            + ".png"
        )
        plt.close()

    # Make predictions with model
    # Specify variables to be used as predictors.
    prediction_grid = rbase.expand_grid(
        NoiseType=rbase.unique(rbase.c(rbase.as_character(r_df.rx2("NoiseType")))),
        Level_dB=rbase.sort(rbase.unique(rbase.c(r_df.rx2(independent_var)))),
    )
    # Make predictions
    predicted_values = rstats.predict(model, prediction_grid, level=0)
    # Combine predictors with predictions
    predictions = rbase.data_frame(prediction_grid, predicted=predicted_values)

    # Posthoc tests by stimulus intensity, corrected for multiple comparisons
    tt = rlsmeans.lstrends(
        model,
        "NoiseType",
        var=independent_var,
        at=rbase.list(
            Level_dB=rbase.sort(rbase.unique(rbase.c(r_df.rx2(independent_var))))
        ),
        by=independent_var,
        transform="response",
    )
    comparisons = rbase.rbind(rgraphics.pairs(tt, by=independent_var), adjust="mvt")
    if print_results:
        print("")
        print(comparisons)

    # Create output variables
    predictions_out = pd.DataFrame(pandas2ri.rpy2py_floatvector(predictions)).astype(
        {"NoiseType": "category"}
    )
    comparisons_out = pandas2ri.rpy2py_dataframe(rbase.summary(comparisons))

    # prt = predictions_out
    # prt['predicted'] = np.exp(prt[['predicted']])
    # print(prt)
    # # Confidence intervals
    # confint = rlsmeans.lsmeans(model, 'NoiseType', var=independent_var,
    #                            at=rbase.list(Level_dB=rbase.sort(rbase.unique(rbase.c(r_df.rx2(independent_var))))),
    #                            by=independent_var, transform='response')
    # print(confint)

    # print(rstats.predict(model, prediction_grid, level=0))

    return predictions_out, comparisons_out, model


def plot_wave1_amplitude_preliminary(df, SPLs=[], xlim=[], ylim=[], LME_SPLs=[]):
    """Temporary function for plotting wave 1. Work in progress, especially considering the LMMs."""

    rstats = rpackages.importr("stats")

    # Set default values
    if len(SPLs) > 0:
        df = df[
            np.all([df["Level_dB"] >= min(SPLs), df["Level_dB"] <= max(SPLs)], axis=0)
        ]
    if len(xlim) == 0:
        xlim = [
            10 * np.floor(min(df["Level_dB"]) / 10) - 2,
            10 * np.ceil(max(df["Level_dB"]) / 10) + 2,
        ]
    if len(ylim) == 0:
        ylim = [
            np.log(100 * np.floor(min(df["W1amp"]) / 100)),
            np.log(100 * np.ceil(max(df["W1amp"]) / 100)),
        ]
    if len(LME_SPLs) == 0:
        LME_SPLs = SPLs

    tmp = np.diff(np.log(ylim))
    y_marker = np.exp([v * tmp for v in [0.94, 0.9, 0.86]] + np.log(min(ylim)))

    alpha = 0.01
    jitter_factor = 1.0

    # Remove outliers!
    df = select_data(df, is_outlier=False)

    # Set figure shape
    uStimFreqs = np.sort(df["Freq_Hz"].unique())
    uABRtimes = df["ABRtime"].unique()
    nRows = len(uStimFreqs)
    nCols = 2
    fig = plt.figure(figsize=[4.8 * nCols, 4.0 * nRows])

    # Figure title
    u_supp = df["supplier"].unique()
    u_nois = df["noiseSPL"].unique()
    u_anal = df["analyzer_ID"].unique()
    u_expe = df["experimenter_ID"].unique()
    if len(u_supp) > 1:
        supp_ID = "+".join(u_supp)
    else:
        supp_ID = u_supp[0]
    if len(u_nois) > 1:
        nois_ID = "+".join(u_nois)
    else:
        nois_ID = u_nois[0]
    if len(u_anal) > 1:
        anal_ID = "+".join(u_anal)
    else:
        anal_ID = u_anal[0]
    if len(u_expe) > 1:
        expe_ID = "+".join(u_expe)
    else:
        expe_ID = u_expe[0]
    fig.suptitle(
        supp_ID
        + ", "
        + df["special"].iloc[0]
        + ", "
        + str(nois_ID)
        + " dB SPL, "
        + [abrt for abrt in uABRtimes if abrt != "baseline"][0]
        + ", α = "
        + str(alpha)
        + "\nanalyzer "
        + anal_ID
        + " , experimenter "
        + expe_ID,
        fontsize=16,
    )

    # Statistics and plotting below
    labelcolor = reference_values()["colormap_alternate"]

    # Loop over different frequencies and times of ABR recording (baseline, 24h, 2w) and only plot 24h and 2w plots
    for count_SF, SF in enumerate(uStimFreqs):
        # Narrow down dataframe to have single frequency
        df_small = select_data(df, Freq_Hz=SF)

        # If only a single value is available for a given SPL, remove it before fitting LME. Otherwise the
        # fit_LME throws an error.
        df_keep = (
            df_small[["Level_dB", "NoiseType", "ID"]]
            .groupby(["Level_dB", "NoiseType"])
            .agg("count")
            > 1
        )
        df_small_LME = (
            df_small.set_index(["Level_dB", "NoiseType"])
            .join(df_keep, rsuffix="_keep")
            .reset_index()
            .astype({"NoiseType": "category"})
        )
        df_small_LME = select_data(df_small_LME, ID_keep=True)

        # Narrow down SPLs for LME, mostly to counter numeric instability when lower SPLs included
        df_small_LME = df_small_LME[
            np.all(
                [
                    df_small_LME["Level_dB"] >= min(LME_SPLs),
                    df_small_LME["Level_dB"] <= max(LME_SPLs),
                ],
                axis=0,
            )
        ]

        # Run LME
        print("Running LME for {f} kHz".format(f=SF))
        predictions, comparisons, model = fit_lme(
            df_small_LME, weighted=True, qqplot=False, print_results=False
        )

        # Plot data seperately for each noise type, e.g., 'baseline', 'NNT', and 'DNT'
        ax = plt.subplot(nRows, nCols, (count_SF + 1) * nCols - 1)
        ax.set_yscale("log")
        sign_masks = {}
        for NT in df["NoiseType"].unique()[::-1]:
            # Narrow down data for current NoiseType
            df_plot = select_data(df_small, NoiseType=NT)

            # Plot data
            jittered_level = np.add(
                df_plot["Level_dB"],
                np.random.uniform(
                    -jitter_factor, jitter_factor, len(df_plot["Level_dB"])
                ),
            )
            plt.plot(
                jittered_level,
                df_plot["W1amp"].values,
                marker="o",
                linestyle="",
                markersize=3,
                markeredgewidth=0.3,
                color=labelcolor[NT],
            )

            # Find range of sound intensities on which LME is based...
            min_level = min(df_plot["Level_dB"])
            max_level = max(df_plot["Level_dB"])
            # ... and create mask for plotting
            predictions_plot = predictions[predictions["NoiseType"] == NT]
            plot_mask = (predictions_plot["Level_dB"] >= min_level) & (
                predictions_plot["Level_dB"] <= max_level
            )

            # Plot model
            plt.plot(
                predictions_plot["Level_dB"][plot_mask],
                np.exp(predictions_plot["predicted"][plot_mask]).values,
                linestyle="-",
                color=labelcolor[NT],
            )

            # Create handles for legend and plot nothing
            plt.plot(
                np.nan,
                np.nan,
                linestyle="-",
                marker="o",
                markersize=3,
                markeredgewidth=0.3,
                color=labelcolor[NT],
                label=NT,
            )

            # Save plot mask for significance markers
            sign_masks[NT] = plot_mask

        # Add markers for significant data
        uContrast = comparisons["contrast"].unique()
        if len(uContrast) <= 3:
            uContrast = np.array(uContrast)[
                np.argsort(
                    [
                        l.lower()
                        for l in [
                            w.replace("ZT15 - ZT3", "ZT3 - ZT15") for w in uContrast
                        ]
                    ]
                )[::-1]
            ]
            for count_contrast, contrast in enumerate(uContrast):
                if (
                    contrast == "ZT15 - ZT3"
                    or contrast == "ZT3 - ZT15"
                    or contrast == "ZT23 - ZT11"
                    or contrast == "ZT11 - ZT23"
                    or contrast == "ZT1 - ZT13"
                    or contrast == "ZT13 - ZT1"
                ):
                    mrkr = "o"
                elif (
                    contrast == "ZT3 - baseline"
                    or contrast == "ZT1 - baseline"
                    or contrast == "ZT23 - baseline"
                ):
                    mrkr = "^"
                elif (
                    contrast == "ZT15 - baseline"
                    or contrast == "ZT13 - baseline"
                    or contrast == "ZT11 - baseline"
                ):
                    mrkr = "s"
                else:
                    raise ValueError("Undefined contrast for significance markers.")
                s_mask = np.all(
                    [v for k, v in sign_masks.items() if k in contrast], axis=0
                )
                comparisons_small = comparisons[comparisons["contrast"] == contrast][
                    s_mask
                ]
                significant_levels = comparisons_small[
                    comparisons_small["p.value"] <= alpha
                ]["Level_dB"].astype("int64")
                plt.plot(
                    significant_levels,
                    np.ones(len(significant_levels)) * y_marker[count_contrast],
                    mec="k",
                    mfc="w",
                    mew=1.5,
                    ms=4,
                    marker=mrkr,
                    linestyle="",
                    label=contrast,
                )

        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.ylabel("Wave 1 amplitude (nV)", fontsize=16)
        if count_SF == 0:
            handles, labels = ax.get_legend_handles_labels()
            labels = [w.replace("ZT15 - ZT3", "ZT3 - ZT15") for w in labels]
            half_length = int(len(handles) / 2)
            first_half_handles = handles[:half_length]
            second_half_handles = handles[half_length:]
            first_half_labels = labels[:half_length]
            second_half_labels = labels[half_length:]
            sort_index_first = np.argsort(first_half_labels)[::-1]
            sort_index_second = np.argsort([l.lower() for l in second_half_labels])[
                ::-1
            ]  # repeats sorting that also happens ~30 lines above, left in in case of edits
            handles = np.hstack(
                [
                    np.array(first_half_handles)[sort_index_first],
                    np.array(second_half_handles)[sort_index_second],
                ]
            )
            labels = np.hstack(
                [
                    np.array(first_half_labels)[sort_index_first],
                    np.array(second_half_labels)[sort_index_second],
                ]
            )
            plt.legend(
                handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.4)
            )  # , loc='center left')
        plt.title(str(SF / 1e3) + " kHz")
        if count_SF == len(uStimFreqs) - 1:
            plt.xlabel("Stimulus intensity (dB SPL)", fontsize=16)

        from matplotlib.ticker import MaxNLocator

        #        import probscale
        plt.subplot(nRows, nCols, (count_SF + 1) * nCols).yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )
        stats.probplot(list(rstats.residuals(model)), dist="norm", plot=plt)
        #        probscale.probplot(list(rstats.residuals(model)), bestfit=True, estimate_ci=True,
        #                           line_kws={'label': 'BF Line', 'color': 'r'},
        #                           scatter_kws={'label': 'Model residuals'},
        #                           problabel='Probability (%)')

        if count_SF != 0:
            plt.title("")
        if count_SF != nRows - 1:
            plt.xlabel("")

    plt.show()
    return


def awfread(path):
    """TDT .awf file reader.

    Parameters
    ----------
    path : string
        String containing the path to the awf file to be imported.

    Returns
    -------
    data : dictionary
        Dictionary containing all data from specified awf file. The waveforms
        can be found in

            data['groups'][i]['wave']

        where i will be in range(0, 30).
    """

    # Initialize parameters
    isRZ = False

    RecHead = dict()
    groups = []
    data = dict()

    with open(path, "rb") as fid:

        # Read RecHead data
        RecHead["nens"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead["ymax"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        RecHead["ymin"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        RecHead["autoscale"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        RecHead["size"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead["gridsize"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead["showgrid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        RecHead["showcur"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        RecHead["TextMargLeft"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead["TextMargTop"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead["TextMargRight"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
        RecHead["TextMargBottom"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

        bFirstPass = True

        for x in range(0, 30):

            # Create dict for looping
            loop_groups = dict()

            loop_groups["recn"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["grpid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            # Read temporary timestamp
            if bFirstPass:
                ttt = np.fromfile(fid, dtype=np.int64, count=1)
                fid.seek(-8, 1)
                # Make sure timestamps make sense.
                if (
                    datetime.now().toordinal()
                    - (ttt / 86400 + date.toordinal(date(1970, 1, 1)))
                    > 0
                ):
                    isRZ = True
                    data["fileTime"] = datetime.fromtimestamp(ttt).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    data["fileType"] = "BioSigRZ"
                else:
                    ttt = np.fromfile(fid, dtype=np.uint32, count=1)
                    data["fileTime"] = datetime.fromtimestamp(ttt).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    fid.seek(-4, 1)
                    data["fileType"] = "BioSigRP"
                bFirstPass = False

            if isRZ:
                loop_groups["grp_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups["grp_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            loop_groups["newgrp"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["sgi"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            # CAREFUL! TROUBLE IN CONVERTING FROM MATLAB CODE.
            # I get a '\01' byte, which translates to "start of heading"
            # Using int8 seems to give the right channel number, but this might be wrong.
            # MATLAB code:
            # groups['chan'].iloc[x] = int16(fread(fid,1,'char'))
            # groups['rtype'].iloc[x] = int16(fread(fid,1,'char'))
            loop_groups["chan"] = np.fromfile(fid, dtype=np.int8, count=1)[0]
            loop_groups["rtype"] = np.fromfile(fid, dtype=np.int8, count=1)[0]

            loop_groups["npts"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["osdel"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["dur"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["srate"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

            loop_groups["arthresh"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["gain"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["accouple"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            loop_groups["navgs"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["narts"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            if isRZ:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            tmp = np.zeros(10)
            for i in range(0, 10):
                tmp[i] = np.fromfile(fid, dtype=np.float32, count=1)
            loop_groups["vars"] = tmp

            cursors = []
            for i in range(0, 10):
                loop_cursors = dict()
                loop_cursors["tmar"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
                loop_cursors["val"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
                tmp_str = fid.read(20).decode("utf-8").split("\0")
                loop_cursors["desc"] = [
                    x for x in tmp_str if x and np.size(tmp_str) > 1
                ]
                loop_cursors["xpos"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors["ypos"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors["hide"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_cursors["lock"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                cursors.append(loop_cursors)
            loop_groups["cursors"] = cursors

            # Open the group
            loop_groups["grpn"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["frecn"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["nrecs"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            tmp_str = fid.read(16).decode("utf-8").split("\0")
            loop_groups["ID"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(16).decode("utf-8").split("\0")
            loop_groups["ref1"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(16).decode("utf-8").split("\0")
            loop_groups["ref2"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(50).decode("utf-8").split("\0")
            loop_groups["memo"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            if isRZ:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int64, count=1)[0]
            else:
                loop_groups["beg_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                loop_groups["end_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            tmp_str = fid.read(100).decode("utf-8").split("\0")
            loop_groups["sgfname1"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(100).decode("utf-8").split("\0")
            loop_groups["sgfname2"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName1"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName2"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName3"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName4"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName5"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName6"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName7"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName8"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName9"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(15).decode("utf-8").split("\0")
            loop_groups["VarName10"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]

            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit1"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit2"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit3"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit4"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit5"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit6"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit7"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit8"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit9"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
            tmp_str = fid.read(5).decode("utf-8").split("\0")
            loop_groups["VarUnit10"] = [
                x for x in tmp_str if x and np.size(tmp_str) > 1
            ]

            loop_groups["SampPer_us"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

            loop_groups["cc_t"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            loop_groups["version"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["postproc"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            tmp_str = fid.read(92).decode("utf-8").split("\0")
            loop_groups["dump"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            loop_groups["bid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["comp"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["x"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["y"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            loop_groups["traceCM"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
            loop_groups["tokenCM"] = np.fromfile(fid, dtype=np.int16, count=1)[0]

            loop_groups["col"] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            loop_groups["curcol"] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            blurb = []
            for i in range(0, 5):
                loop_blurb = dict()
                loop_blurb["type"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["incid"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["hide"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["x"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["y"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                loop_blurb["manplace"] = np.fromfile(fid, dtype=np.int16, count=1)[0]
                tmp_str = fid.read(12).decode("utf-8").split("\0")
                loop_blurb["txt"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]
                blurb.append(loop_blurb)
            loop_groups["blurb"] = blurb
            loop_groups["ymax"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            loop_groups["ymin"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
            tmp_str = fid.read(100).decode("utf-8").split("\0")
            loop_groups["equ"] = [x for x in tmp_str if x and np.size(tmp_str) > 1]

            groups.append(loop_groups)

        for x in range(0, 30):
            if groups[x]["bid"] > 0 and groups[x]["npts"] > 0:
                npts = groups[x]["npts"]
                groups[x]["wave"] = np.fromfile(fid, dtype=np.float32, count=npts)
            else:
                groups[x]["wave"] = []

    data["RecHead"] = RecHead
    data["groups"] = groups

    return data


def load_data_dp(path, print_skipped=False):
    """Load DP data from .awf files and find power of F1, F2, and 2*F1-F2.

    The .awf files are created with BioSigRZ (Tucker-Davis Technologies) during
    experiments. The files are fully loaded, including power spectrum. Then,
    addditional information on primary components and distortion products are
    autodetected.

    Parameters
    ----------
    path : string
        A string indicating the parent directory from which to import all DP
        .awf files from all subdirectories.
    print_skipped : {'True', 'False'}, optional
        If set to True, filenames that are skipped for unclear NoiseType or ABR
        are displayed. The default is False.

    Returns
    -------
    data : array of dict
        Each element of the array is a dictionary of a single .awf file and its
        associated data.
    """

    # Initialize reference arrays
    reference = reference_values()["load_data_dp"]
    exp_ref = reference["experimenter"]
    supplier = reference["supplier"]
    NoiseSPL = reference["NoiseSPL"]
    ABRtimes = reference["ABRtimes"]
    NoiseTypes = reference["NoiseTypes"]

    # Loop over all files in specified directory tree and load all DP .awf files
    all_data = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".awf") and "DP" in file:
                full_file = os.path.join(root, file)

                # Information on skipped files when requested
                if "femaleeffect" in file.lower() or "evi_effect" in root.lower():
                    special = "female_effect"
                elif "sham" in root.lower():
                    special = "sham"
                elif "2hour" in root.lower():
                    special = "2hour"
                else:
                    special = "none"
                temp_NoiseType = NoiseTypes[
                    np.array([x.lower() in file.lower() for x in NoiseTypes])
                ]
                if temp_NoiseType.size == 0:
                    if print_skipped:
                        print(file + " has no readable NoiseType; skipping file.")
                    continue
                temp_ABRtime = ABRtimes[
                    np.array([x.lower() in file.lower() for x in ABRtimes])
                ]
                if temp_ABRtime.size == 0:
                    if print_skipped:
                        print(file + " has no readable ABRtime; skipping file.")
                    continue

                # Experiment information and filename
                temp_supplier = supplier[
                    np.array([x.lower() in root.lower() for x in supplier])
                ]
                temp_NoiseSPL = NoiseSPL[
                    np.array([x.lower() in root.lower() for x in NoiseSPL])
                ]
                file_info = {
                    "file_name": file,
                    "supplier": temp_supplier[0],
                    "special": special,
                    "NoiseSPL": temp_NoiseSPL[0],
                    "NoiseType": temp_NoiseType[0],
                    "ABRtime": temp_ABRtime[0],
                }

                first_char = file.split("_")[1][0].upper()
                if (
                    first_char != "C"
                    and first_char != "E"
                    and first_char != "I"
                    and first_char != "B"
                    and first_char != "X"
                ):
                    tmpID = (
                        "C"
                        + file.split("_")[1]
                        + "_"
                        + file.split("_")[2].split(" ")[0]
                    )
                else:
                    tmpID = file.split("_")[1] + "_" + file.split("_")[2].split(" ")[0]
                file_info["ID"] = tmpID

                experimenter = next(
                    (key for key, value in exp_ref.items() if tmpID in value), None
                )
                if experimenter == None:
                    raise LookupError(
                        "Undefined experimenter for animal "
                        + tmpID
                        + ". Add values for file "
                        + file
                        + " in function reference_values()."
                    )
                file_info["experimenter_ID"] = experimenter

                # Actual loading of data
                awfdata = awfread(full_file)
                awfdata["fileInfo"] = file_info

                # Detect peaks in spectrum
                awfdata = get_peaks(awfdata)

                # Add to output argument
                all_data.append(awfdata)

    return all_data
