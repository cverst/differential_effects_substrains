import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from utils.plotting_stats import plot_wave1_distribution

FIELDS_RLM = [
    "file_number",
    "level_db",
    "rlm",
    "rlm_error",
    "rlm_error_standardized",
    "standardization_std",
]


def abr_prep(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper function for preprocessing of ABR data.

    Args:
        df (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """

    print("Detecting outliers...")
    df = detect_outliers_abr(df)

    print("Averaging and removing duplicates...")
    df = duplicates_to_means(df)
    
    print("Done.")

    plot_wave1_distribution(df)

    return df


def detect_outliers_abr(df: pd.DataFrame) -> pd.DataFrame:
    """Wave 1 outlier detection for ABR data.

    This functin detects outliers in four steps:
    1. A robust linear model is fit to wave 1 data
    2. The data is standardized (normalization to mean and standard deviation)
    3. A skewed normal distribution is fit to those data
    4. Values outside the 95% confidence interval are indicated as outlier
    
    Args:
        df (pd.DataFrame): Dataframe with all wave 1 recordings to consider for
        outlier detection.

    Returns:
        pd.DataFrame: Original dataframe with several new columns of
            information on the outlier detection procedure. Column name
            "is_outlier" is a boolean indicating if a data point is an outlier.
    """

    CONFINT = .95

    file_nums = df.loc[:, "file_number"].unique()

    df_rlm = pd.DataFrame(columns=FIELDS_RLM)
    # First construct dataframe with all RLM values, then merge
    for fn in file_nums:
        df_rlm = df_rlm.append(wave1_rlm(df.query("file_number == @fn")))

    df = df.merge(df_rlm, how="left", on=["file_number", "level_db"])

    # Outlier detection
    standardized_deviations = df.loc[:, "rlm_error_standardized"].dropna()  # remove NaNs where RLM was impossible
    distrib_params = stats.skewnorm.fit(standardized_deviations)  # a, loc, scale
    interval = stats.skewnorm.interval(CONFINT, *distrib_params)
    df.loc[:, "is_outlier"] = np.any(
        [
            df.loc[:, "rlm_error_standardized"] < interval[0],
            df.loc[:, "rlm_error_standardized"] > interval[1],
        ],
        axis=0,
    )
    df.loc[:, "confint_low"] = (interval[0] * df.loc[:, "standardization_std"]) + df.loc[:, "rlm"]
    df.loc[:, "confint_high"] = (interval[1] * df.loc[:, "standardization_std"]) + df.loc[:, "rlm"]

    return df


def wave1_rlm(df: pd.DataFrame) -> pd.DataFrame:
    """Fit robust linear model to wave 1 data

    The model is fit to the input-output relation of the data. The input-output
    relation is near linear on a logarithmic scale:

        (log(wave 1 amplitude) vs. stimulus level)
    
    The wave 1 data must come from a single experiment, meaning that it must
    have a single file number.
        

    Args:
        df (pd.DataFrame): Dataframe with wave 1 data from single file number,
            i.e., experiment.

    Returns:
        pd.DataFrame: New dataframe with RLM information.
    """

    df = df.copy()
    assert df.loc[:, "file_number"].nunique() == 1, "Wave 1 amplitudes not from a single file!"

    wave1_log = np.log(df.loc[:, "wave1_amp"])
    if df.shape[0] > 2:
        x = df.loc[:, "level_db"].values
        df.loc[:, "rlm"] = rlm_fit(x, wave1_log)
    else:
        df.loc[:, "rlm"] = np.nan
    
    df.loc[:, "rlm_error"] = wave1_log - df.loc[:, "rlm"]

    # Use unweighted std for standardization. Weighted std is numerically
    # unstable, because the average of the traces is very close to zero.
    std_temp = np.std(df.loc[:, "rlm_error"])
    df.loc[:, "rlm_error_standardized"] = df.loc[:, "rlm_error"] / std_temp
    df.loc[:, "standardization_std"] = std_temp
    # df_temp["W1amp_normalized"] = (wave1_log - np.mean(wave1_log)) / np.std(wave1_log)

    df.astype({
        "file_number": "int64",
        "level_db": "float64",
        "rlm": "float64",
        "rlm_error": "float64",
        "rlm_error_standardized": "float64",
        "standardization_std": "float64",
    })

    return df.loc[:, FIELDS_RLM]


def rlm_fit(x: np.array, y: np.array) -> list:
    """Fit robust regression model
    
    Fit statsmodels.api.RLM to input data. Robust regression is insensitive to
    outliers.
    
    Args:
        x (np.array): Independent parameter.
        y (np.array): Dependent parameter.

    Returns:
        list: Predicted value of y at x.
    """

    # Add intercept
    x = sm.add_constant(x)  

    # Fit model
    model = sm.RLM(y, x)
    results = model.fit()

    return results.fittedvalues


def duplicates_to_means(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicates from wave 1 data and replaces them with their mean

    Args:
        df (pd.DataFrame): Dataframe with suspected dusplicates.

    Returns:
        pd.DataFrame: Original dataframe with duplicates removed and their
            means added.
    """

    # Preserve datatypes by first storing them
    df_dtypes = df.dtypes
    fields_fixed = [
        "substrain",
        "id",
        "noise_spl",
        "experimenter_id",
        "noise_type",
        "abr_time",
        "level_db",
        "freq_hz",
    ]
    # TODO: Find alternative to averaging std and confint high/low
    fields_mean = [
        "wave1_amp",
        "threshold",
        "rlm",
        "rlm_error",
        "rlm_error_standardized",
        "standardization_std",
        "confint_low",
        "confint_high",
    ]

    n_dupes = df.duplicated(subset=fields_fixed).sum()
    if n_dupes == 0:
        print("  No duplicates found.")
        return df
    else:
        print(f"  Found {n_dupes} duplicates.")

    # Merge duplicates by rather difficult to interpret way
    # TODO: Simplify (pd.merge, etc.).
    duplicates_merged = pd.concat(
        [
            pd.concat(
                [
                    g[fields_fixed].iloc[0],
                    np.mean(g[fields_mean]),
                    pd.Series(
                        {
                            "file_number": np.nan,
                            "analyzer_id": "|"
                            + "+".join(g.loc[:, "analyzer_id"].unique())
                            + "|",
                            "is_outlier": any(g.loc[:, "is_outlier"]),
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
    df_dtypes.loc["analyzer_id"] = pd.api.types.CategoricalDtype(
        categories=df.loc[:, "analyzer_id"].unique()
    )
    df = df.astype(df_dtypes)

    return df
