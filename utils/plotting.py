import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import rpy2.robjects.packages as rpackages
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgba
from utils.statistics import fit_lme, nonparametric_tests


COLORMAP = {
    "baseline": [0.4, 0.4, 0.4],
    "ZT3": "r",
    "ZT15": "b",
}
PALE_COLORMAP = {
    "baseline": [0.7, 0.7, 0.7],
    "ZT3": [1, 0.6, 0.6],
    "ZT15": [0.6, 0.6, 1],
}


def wave1_distribution(df: pd.DataFrame) -> None:
    """Plot distribution of standardized wave 1 data used for outlier detection

    Args:
        df (pd.DataFrame): Dataframe of wave 1 data after outlier detection.
    """

    N_BINS = 256
    CONFINT = 0.95

    # Remove NaNs
    standardized_deviations = df.loc[
        :, "rlm_error_standardized"
    ].dropna()  # remove NaNs where RLM was impossible

    # Calculate histogram
    hist, _ = np.histogram(
        standardized_deviations, bins=N_BINS, range=(-5, 5), density=True
    )

    plt.figure(figsize=[14, 4])

    # Plot the distribution
    plt.subplot(1, 2, 1)
    x = np.linspace(-5, 5, num=N_BINS)
    plt.bar(x, hist, width=0.08)
    plt.xlabel("Standardized wave 1 amplitude")
    plt.ylabel("Relative number of occurences")
    plt.title("Wave 1 amplitude distribution and PDF")

    # Plot PDF
    distrib_params = stats.skewnorm.fit(standardized_deviations)  # a, loc, scale
    plt.plot(x, stats.skewnorm.pdf(x, *distrib_params), "r", lw=2)
    interval = stats.skewnorm.interval(CONFINT, *distrib_params)
    plt.plot(interval, [0.01, 0.01], "y", lw=10)
    plt.legend(
        ["PDF", f"{round(CONFINT*100)}% confidence interval", "Wave 1 amplitude"],
        loc="upper left",
    )

    # Count number of outliers
    n_outliers = (
        df.astype({"level_db": "category"})
        .groupby(by=["level_db", "is_outlier"])
        .agg("count")["id"]
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


def abr_io(df: pd.DataFrame) -> None:
    """Plot input-output functions indicate outliers.

    This function plots the relationship between wave 1 amplitude and stimulus
    level for all recordings (baseline, 24 hour, 2 week) of a single subject
    ("id"), separated by stimulus frequency.

    Args:
        df (pd.DataFrame): Dataframe of wave 1 data after outlier detection.
            The dataframe must Make sure only a single experiment is parsed.
    """

    assert (
        df.loc[:, "id"].nunique() == 1
    ), "More than 1 'id' present in input dataframe!"

    # Get abr_times and stim_freqs and order them
    unique_abr_times = df.loc[:, "abr_time"].unique()
    unique_abr_times = np.sort([uat[::-1] for uat in unique_abr_times])
    unique_abr_times = [uat[::-1] for uat in unique_abr_times]
    unique_stim_freqs = np.sort(df.loc[:, "freq_hz"].unique())

    # Determine shape of figure
    nrows = len(unique_abr_times)  # row for each abr_time
    ncols = len(unique_stim_freqs)  # column for each stim_freq
    plt.figure(figsize=[3.2 * ncols, 3 * nrows])

    # Plotting
    for count_abrt, abrt in enumerate(unique_abr_times):
        for count_sf, sf in enumerate(unique_stim_freqs):

            # Narrow down dataframe
            df_temp = df.query("abr_time == @abrt & freq_hz == @sf")

            if len(df_temp) > 0:

                # Extract plot data
                x = df_temp["level_db"]
                y = np.log(df_temp["wave1_amp"])
                mask = df_temp["is_outlier"]

                # Plotting
                plt.subplot(nrows, ncols, count_abrt * ncols + count_sf + 1)
                plt.plot(x, y, marker="o", linestyle="--", color="b")
                plt.plot(
                    x[mask],
                    y[mask],
                    marker="o",
                    markeredgecolor="r",
                    markerfacecolor="r",
                    linestyle="",
                )
                plt.plot(x, df_temp["rlm"], "k:")

                plt.plot(x, df_temp["confint_low"], "y:")
                plt.plot(x, df_temp["confint_high"], "y:")

                # Labeling
                plt.xlabel("Stimulus level (dB SPL)")
                plt.title(
                    df_temp["id"].iloc[0]
                    + ", "
                    + df_temp["noise_type"].iloc[0]
                    + ", "
                    + str(round(sf / 1e3))
                    + " kHz"
                )
                if count_sf == 0:
                    plt.ylabel("log(wave 1 amplitude) re 1 nV")

    plt.tight_layout()
    plt.show()

    return


def abr_threshold(df: pd.DataFrame, ylim: list = [-5, 95]) -> None:
    """Plot ABR thresholds and check statistical significance

    This function first plots different noisetypes within each
    frequency. Subsequently it calls the function
    nonparametric_tests for statistical evaluation.

    Parameters
    ----------
    df : pandas.DataFrame


    Args:
        df (pd.DataFrame): A selection of ABR data used for threshold analysis.
        ylim (list, optional): Limits of Y-axis. Defaults to [-5, 95].
    """

    JITTER_FACTOR = 0.05

    unique_abr_times = sorted(df.loc[:, "abr_time"].unique())  # baseline is now last

    # Only keep rows that are threshold
    colnames = [
        "substrain",
        "id",
        "noise_spl",
        "analyzer_id",
        "experimenter_id",
        "noise_type",
        "abr_time",
        "freq_hz",
        "threshold",
    ]
    df = df[colnames].groupby(colnames[:-1]).agg(pd.Series.unique)

    df = df.dropna().reset_index()

    # Create figure
    fig = plt.figure()

    # Figure title
    u_subs = df.loc[:, "substrain"].unique()
    u_nois = df.loc[:, "noise_spl"].unique()
    u_abrt = unique_abr_times
    u_anal = df.loc[:, "analyzer_id"].unique()
    u_expe = df.loc[:, "experimenter_id"].unique()
    if len(u_subs) > 1:
        subs_id = "+".join(u_subs)
    else:
        subs_id = u_subs[0]
    if len(u_nois) > 1:
        nois_id = "+".join(str(e) + " dB" for e in u_nois)
    else:
        nois_id = u_nois[0]
    if len(unique_abr_times) > 1:
        abrt_id = "+".join(u_abrt)
    else:
        abrt_id = u_abrt[0]
    if len(u_anal) > 1:
        anal_id = "+".join(u_anal)
    else:
        anal_id = u_anal[0]
    if len(u_expe) > 1:
        expe_id = "+".join(u_expe)
    else:
        expe_id = u_expe[0]
    fig.suptitle(
        subs_id
        + ", "
        + str(nois_id)
        + " SPL, "
        + abrt_id
        + "\n"
        + "analyzer "
        + anal_id
        + ", experimenter "
        + expe_id,
        fontsize=14,
    )

    df.loc[:, "freq_hz"] /= 1e3
    df.loc[:, "noise_type"] = df.loc[:, "noise_type"].astype(str)
    df = df.sort_values(by=["noise_type"], ascending=False)

    # Plotting
    df.loc[:, "threshold"] = np.add(
        df.loc[:, "threshold"],
        np.random.uniform(-JITTER_FACTOR, JITTER_FACTOR, len(df.loc[:, "threshold"])),
    )

    ax = plt.axes()
    df.loc[:, "id"] = df.loc[:, "id"].astype(str)
    df.loc[:, "freq_hz_cat"] = df.loc[:, "freq_hz"].astype("category")
    df.loc[:, "freq_hz_cat"] = df.loc[:, "freq_hz_cat"].cat.rename_categories(
        {8.0: "8", 12.0: "12", 16.0: "16", 24.0: "24", 32.0: "32"}
    )

    sns.lineplot(
        ax=ax,
        x="freq_hz_cat",
        y="threshold",
        hue="noise_type",
        units="id",
        data=df,
        estimator=None,
        linewidth=0.5,
        mew=0.5,
        mfc="w",
        marker="o",
        ms=4,
        palette=PALE_COLORMAP,
        legend=False,
    )

    clrs = [plt.getp(line, "color") + [1.0] for line in ax.lines]
    for c, v in enumerate(ax.lines):
        plt.setp(v, markeredgecolor=clrs[c])
    n_lines_light = len(ax.lines)

    mean_sem = (
        df.loc[:, ["noise_type", "freq_hz", "threshold"]]
        .groupby(by=["noise_type", "freq_hz"])
        .agg(["mean", "sem"])
        .reset_index()
    )
    for nt in sorted(mean_sem.loc[:, ("noise_type", "")].unique(), reverse=True):
        small_df = mean_sem.query("@mean_sem.noise_type == @nt")
        ax.errorbar(
            np.arange(0, len(small_df)),
            small_df["threshold"]["mean"],
            yerr=small_df["threshold"]["sem"],
            color=COLORMAP[nt],
            linewidth=1.5,
            capsize=4,
            marker="o",
            markersize=6,
            zorder=n_lines_light + 1,
            label=nt,
        )

    plt.ylim(ylim)

    ax.legend()

    plt.xlabel("Stimulus frequency (kHz)", fontsize=14)
    plt.ylabel("Threshold (dB SPL)", fontsize=14)
    plt.show()

    # Statistical tests
    nonparametric_tests(df)

    plt.show()

    return


def wave1_amplitude(
    df: pd.DataFrame,
    stim_levels: list = [],
    xlim: list = [],
    ylim: list = [],
    lme_stim_levels: list = [],
) -> None:
    """Plot wave 1 data and fit LME model

    Visualize wave 1 amplitude data and indicate where different experimental
    groups are significantly different.
    1. Each row of panels corresponds to a different stimulus frequency
    2. The left column shows raw data and LME results. The legend indicates
        which group belongs to what color, and which comparison belongs to what
        marker type.
    3. The right column shows probability plots. The linear mized effects model
        assumes that the model residuals are homoskedastic. If that is the
        case, the residuals (blue markers) will follow the red line. These
        plots assist in visual evaluation of this assumption, and their
        interpretation is supported by the result of Levene's test, printed
        above the figure. Neither method alone is fool proof, and residuals
        need to be reasonably homoskedastic.

    Args:
        df (pd.DataFrame): Input dataframe with wave 1 data.
        stim_levels (list, optional): Minimum and maximum stimulus level of the
            data. Any data points outside this range will be removed at the
            start of this function. Defaults to [].
        xlim (list, optional): X-axis limits. Defaults to [].
        ylim (list, optional): Y-axis limits. Defaults to [].
        lme_stim_levels (list, optional): Stimulus levels to consider for
            fitting of LME. The model cannot handle single values for a given
            sound level. The LME analysis can therefore be limited in sound
            level range. Defaults to [].
    """

    rstats = rpackages.importr("stats")

    # Set default values
    if len(stim_levels) > 0:
        df = df.query(f"{min(stim_levels)} <= level_db <= {max(stim_levels)}")
    if len(xlim) == 0:
        xlim = [
            10 * np.floor(min(df.loc[:, "level_db"]) / 10) - 2,
            10 * np.ceil(max(df.loc[:, "level_db"]) / 10) + 2,
        ]
    if len(ylim) == 0:
        ylim = [
            np.log(100 * np.floor(min(df.loc[:, "wave1_amp"]) / 100)),
            np.log(100 * np.ceil(max(df.loc[:, "wave1_amp"]) / 100)),
        ]
    if len(lme_stim_levels) == 0:
        lme_stim_levels = stim_levels

    ALPHA = 0.01
    JITTER_FACTOR = 1.0

    # Calculate y-position of significance markers
    y_range = np.diff(np.log(ylim))
    y_marker = np.exp([v * y_range for v in [0.94, 0.9, 0.86]] + np.log(min(ylim)))

    # Remove outliers!
    df = df.query("is_outlier == False")

    # Set figure shape
    unique_stim_freqs = np.sort(df.loc[:, "freq_hz"].unique())
    unique_abr_times = df.loc[:, "abr_time"].unique()
    n_rows = len(unique_stim_freqs)
    n_cols = 2
    fig = plt.figure(figsize=[4.8 * n_cols, 4.0 * n_rows])

    # Figure title
    u_subs = df["substrain"].unique()
    u_nois = df["noise_spl"].unique()
    u_anal = df["analyzer_id"].unique()
    u_expe = df["experimenter_id"].unique()
    if len(u_subs) > 1:
        subs_id = "+".join(u_subs)
    else:
        subs_id = u_subs[0]
    if len(u_nois) > 1:
        nois_id = "+".join(u_nois)
    else:
        nois_id = u_nois[0]
    if len(u_anal) > 1:
        anal_id = "+".join(u_anal)
    else:
        anal_id = u_anal[0]
    if len(u_expe) > 1:
        expe_id = "+".join(u_expe)
    else:
        expe_id = u_expe[0]
    fig.suptitle(
        subs_id
        + ", "
        + str(nois_id)
        + " dB SPL, "
        + [abrt for abrt in unique_abr_times if abrt != "baseline"][0]
        + ", alpha = "
        + str(ALPHA)
        + "\nanalyzer "
        + anal_id
        + " , experimenter "
        + expe_id,
        fontsize=14,
    )

    # Statistics and plotting below

    # Loop over different frequencies and times of ABR recording (baseline,
    # 24h, 2w) and plot 24h and 2w plots only
    for count_sf, sf in enumerate(unique_stim_freqs):
        # Narrow down dataframe to have single frequency
        df_small = df.query("freq_hz == @sf")

        # If only a single value is available for a given SPL, remove it
        # before fitting LME. Otherwise fit_lme throws an error.
        df_keep = (
            df_small.loc[:, ["level_db", "noise_type", "id"]]
            .groupby(["level_db", "noise_type"])
            .agg("count")
            > 1
        )
        df_small_lme = (
            df_small.set_index(["level_db", "noise_type"])
            .join(df_keep, rsuffix="_keep")
            .reset_index()
            .astype({"noise_type": "category"})
        )
        df_small_lme = df_small_lme.query("id_keep == True")

        # Narrow down stim levels for LME, mostly to counter numeric
        # instability when lower levels included
        df_small_lme = df_small_lme.query(
            f"{min(lme_stim_levels)} <= level_db <= {max(lme_stim_levels)}"
        )

        # Run LME
        print("Running LME for {f} kHz".format(f=sf))
        predictions, comparisons, model = fit_lme(
            df_small_lme, weighted=True, print_results=False
        )

        # Plot data seperately for each noise type, e.g., 'baseline', 'ZT3', and 'ZT15'
        ax = plt.subplot(n_rows, n_cols, (count_sf + 1) * n_cols - 1)
        ax.set_yscale("log")
        signif_masks = {}
        for nt in df.loc[:, "noise_type"].unique()[::-1]:
            # Narrow down data for current NoiseType
            df_plot = df_small.query("noise_type==@nt")

            # Plot data
            jittered_level = np.add(
                df_plot.loc[:, "level_db"],
                np.random.uniform(
                    -JITTER_FACTOR, JITTER_FACTOR, len(df_plot.loc[:, "level_db"])
                ),
            )
            plt.plot(
                jittered_level,
                df_plot["wave1_amp"].values,
                marker="o",
                linestyle="",
                markersize=3,
                markeredgewidth=0.3,
                color=COLORMAP[nt],
            )

            # Find range of sound intensities on which LME is based...
            min_level = min(df_plot.loc[:, "level_db"])
            max_level = max(df_plot.loc[:, "level_db"])
            # ... and create mask for plotting
            predictions_plot = predictions.query("noise_type == @nt")
            plot_mask = (predictions_plot.loc[:, "level_db"] >= min_level) & (
                predictions_plot.loc[:, "level_db"] <= max_level
            )

            # Plot model
            plt.plot(
                predictions_plot.loc[:, "level_db"][plot_mask],
                np.exp(predictions_plot.loc[:, "predicted"][plot_mask]).values,
                linestyle="-",
                color=COLORMAP[nt],
            )

            # Create handles for legend and plot nothing
            plt.plot(
                np.nan,
                np.nan,
                linestyle="-",
                marker="o",
                markersize=3,
                markeredgewidth=0.3,
                color=COLORMAP[nt],
                label=nt,
            )

            # Save plot mask for significance markers
            signif_masks[nt] = plot_mask

        # Add markers for significant data
        unique_contrast = comparisons.loc[:, "contrast"].unique()
        if len(unique_contrast) <= 3:
            unique_contrast = np.array(unique_contrast)[
                np.argsort(
                    [
                        l.lower()
                        for l in [
                            w.replace("ZT15 - ZT3", "ZT3 - ZT15")
                            for w in unique_contrast
                        ]
                    ]
                )[::-1]
            ]
            for count_contrast, contrast in enumerate(unique_contrast):
                if contrast == "ZT15 - ZT3" or contrast == "ZT3 - ZT15":
                    mrkr = "o"
                elif contrast == "ZT3 - baseline":
                    mrkr = "^"
                elif contrast == "ZT15 - baseline":
                    mrkr = "s"
                else:
                    raise ValueError("Undefined contrast for significance markers.")
                s_mask = np.all(
                    [v for k, v in signif_masks.items() if k in contrast], axis=0
                )
                comparisons_small = comparisons.query("contrast == @contrast")[s_mask]
                significant_levels = comparisons_small.loc[
                    comparisons_small.loc[:, "p.value"] <= ALPHA, "level_db"
                ].astype("int64")
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

        plt.ylabel("Wave 1 amplitude (nV)", fontsize=14)

        # Create legend
        if count_sf == 0:
            handles, labels = ax.get_legend_handles_labels()
            labels = [w.replace("ZT15 - ZT3", "ZT3 - ZT15") for w in labels]
            half_length = int(len(handles) / 2)
            first_half_handles = handles[:half_length]
            second_half_handles = handles[half_length:]
            first_half_labels = labels[:half_length]
            second_half_labels = labels[half_length:]
            sort_index_first = np.argsort(first_half_labels)[::-1]
            # TODO: Remove following line?
            sort_index_second = np.argsort([l.lower() for l in second_half_labels])[
                ::-1
            ]  # repeats sorting that also happens earlier in this function, left in in case of edits
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
            )

        plt.title(str(sf / 1e3) + " kHz")
        if count_sf == len(unique_stim_freqs) - 1:
            plt.xlabel("Stimulus intensity (dB SPL)", fontsize=14)

        # Create qqplot / probability plot
        plt.subplot(n_rows, n_cols, (count_sf + 1) * n_cols).yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )
        stats.probplot(list(rstats.residuals(model)), dist="norm", plot=plt)
        plt.gca().xaxis.label.set_size(14)
        plt.gca().yaxis.label.set_size(14)

        if count_sf != 0:
            plt.title("")
        if count_sf != n_rows - 1:
            plt.xlabel("")

    plt.show()
    return


def lvl_diff(df: pd.DataFrame) -> None:

    # Cutoffs for level_f1 - level_f2
    LVL_DIFF_PRIMARIES = 10
    LVL_DIFF_ALLOWED = 20

    plt.figure()

    # With outliers
    _, bins, _ = plt.hist(
        df.loc[:, "level_f1"] - df.loc[:, "level_f2"],
        bins=32,
        label="With outliers",
    )

    # Without outliers
    df_no_outliers = df.query("is_outlier == False")
    plt.hist(
        df_no_outliers.loc[:, "level_f1"] - df_no_outliers.loc[:, "level_f2"],
        bins=bins,
        label="Without outliers",
    )

    ylim = plt.gca().get_ylim()
    x_vlines = [
        LVL_DIFF_PRIMARIES - LVL_DIFF_ALLOWED,
        LVL_DIFF_PRIMARIES + LVL_DIFF_ALLOWED,
    ]
    plt.vlines(x_vlines, ylim[0], ylim[1], colors="r", label="Outlier boundary")

    plt.xlabel("Recorded level 1 - level 2 (dB)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Before (top) and after (bottom) outlier removal")
    plt.show()
    return


def dp_threshold(df: pd.DataFrame, ylim: list) -> None:
    """Create strip plot showing distribution DPOAE threshold

    When parsing a list of dictionaries from .awf files this function does the
    following:

        1) Find the first stimulus intensity that evokes a DP of -5 dB SPL or more and if
        none are found, use 85 dB SPL as threshold.
        2) Plot a strip plot for these thresholds. Separate each Frequency, and
        accept groups of noisetypes within each frequency.
        3) Do a Kruskal-Wallis test (nonparametric) to find differences within frequency
        groups and if a difference is found, do a post hoc analysis between noisetypes
        within a frequency group.

    If input argument "shift=True" the threshold shift between different recording times
    (baseline, 24h, 2w) is plotted and tested.

    Parameters
    ----------
    datadict : list of dictionaries
        A (list of) dictionaries created with load_data_dp.
    """

    NOISEFLOOR = {8: -5, 12: -5, 16: 0, 24: 0, 32: 5}

    # Remove outliers!
    df = df.query("is_outlier == False").copy()

    df.loc[:, "freq_hz"] = df.loc[:, "freq_hz"] / 1e3

    # Create axes
    fig = plt.figure()

    # Figure title
    u_subs = df.loc[:, "substrain"].unique()
    u_nois = df.loc[:, "noise_spl"].unique()
    u_expe = df.loc[:, "experimenter_id"].unique()
    if len(u_subs) > 1:
        subs_id = "+".join(u_subs)
    else:
        subs_id = u_subs[0]
    if len(u_nois) > 1:
        nois_id = "+".join(u_nois)
    else:
        nois_id = u_nois[0]
    if len(u_expe) > 1:
        expe_id = "+".join(u_expe)
    else:
        expe_id = u_expe[0]
    fig.suptitle(
        subs_id + ", " + str(nois_id) + " SPL, experimenter " + expe_id,
        fontsize=14,
    )

    df.loc[:, "above_nf"] = df.apply(
        lambda row: row["level_distprod"] >= NOISEFLOOR[row["freq_hz"]], axis=1
    )

    df_85 = df.query("level_db == 80").copy()
    df_85.loc[:, "level_db"] = 85
    df_85.loc[:, "above_nf"] = True
    df = df.append(df_85).reset_index()

    df_threshold = df.query("above_nf == True").groupby(by=["file_number"]).min()
    df_threshold.loc[:, "threshold"] = df_threshold.loc[:, "level_db"]

    # Plotting
    jitter_factor = 1
    df_threshold.loc[:, "threshold"] = np.add(
        df_threshold.loc[:, "threshold"],
        np.random.uniform(-jitter_factor, jitter_factor, len(df_threshold)),
    )

    ax = plt.axes()
    df_threshold.loc[:, "id"] = df_threshold.loc[:, "id"].astype(str)
    df_threshold.loc[:, "freq_hz_cat"] = df_threshold.loc[:, "freq_hz"].astype(
        "category"
    )
    df_threshold.loc[:, "freq_hz_cat"] = df_threshold.loc[
        :, "freq_hz_cat"
    ].cat.rename_categories({8.0: "8", 12.0: "12", 16.0: "16", 24.0: "24", 32.0: "32"})

    sns.lineplot(
        ax=ax,
        x="freq_hz_cat",
        y="threshold",
        hue="noise_type",
        units="id",
        data=df_threshold,
        estimator=None,
        linewidth=0.5,
        mew=0.5,
        mfc="w",
        marker="o",
        ms=4,
        palette=PALE_COLORMAP,
        legend=False,
    )

    clrs = [plt.getp(line, "color") + [1.0] for line in ax.lines]
    for c, v in enumerate(ax.lines):
        plt.setp(v, markeredgecolor=clrs[c])
    n_lines_light = len(ax.lines)

    mean_sem = (
        df_threshold.loc[:, ["noise_type", "freq_hz", "threshold"]]
        .groupby(by=["noise_type", "freq_hz"])
        .agg(["mean", "sem"])
        .reset_index()
    )
    for nt in sorted(mean_sem.loc[:, ("noise_type", "")].unique(), reverse=True):
        small_df = mean_sem.query("@mean_sem.noise_type == @nt")
        ax.errorbar(
            np.arange(0, len(small_df)),
            small_df.loc[:, ("threshold", "mean")],
            yerr=small_df.loc[:, ("threshold", "sem")],
            color=COLORMAP[nt],
            linewidth=1.5,
            capsize=4,
            marker="o",
            markersize=6,
            zorder=n_lines_light + 1,
            label=nt,
        )

    plt.ylim(ylim)
    ax.legend()
    plt.xlabel("Stimulus frequency F2 (kHz)", fontsize=14)
    plt.ylabel("L1 at threshold (dB SPL)", fontsize=14)
    plt.show()

    # Statistical tests
    nonparametric_tests(df_threshold)

    return


def histology_counts(df: pd.DataFrame) -> None:
    """Plots histological counts.

    Args:
        df (pd.DataFrame): A dataframe with counts of ribbon synapses or counts
            of synapse pairings.
    """

    df.loc[:, "freq_cat"] = df.loc[:, "freq_hz"].astype("category")
    df.loc[:, "freq_cat"] = df.loc[:, "freq_cat"].cat.rename_categories(
        {
            6e3: "6",
            8e3: "8",
            12e3: "12",
            16e3: "16",
            24e3: "24",
            32e3: "32",
            48e3: "48",
            64e3: "64",
        }
    )

    g = sns.relplot(
        x="freq_cat",
        y="counts",
        hue="abr_time",
        units="id",
        estimator=None,
        row="substrain",
        col="noise_spl",
        hue_order=["baseline", "ZT3", "ZT15"],
        row_order=["CaJ", "JRj", "Sca"],
        col_order=[100, 103, 105],
        data=df,
        kind="line",
        lw=0.5,
        mew=0.5,
        mfc="w",
        marker="o",
        ms=4,
        palette=PALE_COLORMAP,
        height=2.2,
        aspect=1.5,
        legend=False,
    )

    mean_sem = (
        df.loc[:, ["substrain", "noise_spl", "abr_time", "freq_hz", "counts"]]
        .groupby(by=["substrain", "noise_spl", "abr_time", "freq_hz"])
        .agg(["mean", "sem"])
        .reset_index()
    )
    for ax in g.axes.flatten():
        clrs = [to_rgba(line.get_color()) for line in ax.lines]
        for c, v in enumerate(ax.lines):
            v.set_markeredgecolor(clrs[c])

        if "CaJ" in ax.get_title():
            small_df = mean_sem.query("@mean_sem.substrain == 'CaJ'")
        elif "JRj" in ax.get_title():
            small_df = mean_sem.query("@mean_sem.substrain == 'JRj'")
        elif "Sca" in ax.get_title():
            small_df = mean_sem.query("@mean_sem.substrain == 'Sca'")
        if "100" in ax.get_title():
            small_df = small_df.query("@mean_sem.noise_spl == 100")
        elif "103" in ax.get_title():
            small_df = small_df.query("@mean_sem.noise_spl == 103")
        elif "105" in ax.get_title():
            small_df = small_df.query("@mean_sem.noise_spl == 105")
        for c, time in enumerate(["ZT3", "ZT15"]):
            plot_df = small_df.query("@mean_sem.abr_time == @time")
            ax.errorbar(
                np.arange(0, len(plot_df)),
                plot_df.loc[:, ("counts", "mean")],
                yerr=plot_df.loc[:, ("counts", "sem")],
                linewidth=1.5,
                capsize=4,
                marker="o",
                markersize=6,
                zorder=1e3,
                label=time,
                color=COLORMAP[time],
            )

    g.set(ylim=(-1, 31), xlabel="Frequency (kHz)", ylabel="Count per IHC")

    ax.legend()
    plt.show()

    return
