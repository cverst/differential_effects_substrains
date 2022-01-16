import numpy as np
import pandas as pd
from scipy import stats
import itertools
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter


# Import R packages
rbase = rpackages.importr("base")
rgraphics = rpackages.importr("graphics")
rstats = rpackages.importr("stats")
rnlme = rpackages.importr("nlme")
rcar = rpackages.importr("car")
rlsmeans = rpackages.importr("lsmeans")


def nonparametric_tests(df: pd.DataFrame) -> None:
    """Performs nonparamametric tests on threshold data.

    This function performs nonparametric tests for thresholds in two subsequent
    steps:
    1) Differences within frequency groups (Kruskal-Wallis
        test (nonparametric))
    2) Differences between noisetypes within a frequency
        group (post hoc Mann-Whitney U)

    The results are printed to the screen.

    Args:
        df (pd.DataFrame): Dataframe with ABR or DP data.
    """

    ALPHA = 0.05

    print("alpha = " + str(ALPHA) + "\n")
    for freq_hz in np.sort(df.loc[:, "freq_hz"].unique()):
        small_df = df.query("freq_hz == {f}".format(f=freq_hz))
        print("Frequency: " + str(freq_hz) + " kHz")
        kw = stats.kruskal(
            *(
                small_df.loc[:, "threshold"][small_df.loc[:, "noise_type"] == nt]
                for nt in small_df.loc[:, "noise_type"].unique()
            )
        )
        print("  Overall Kruskal-Wallis:")
        if kw.pvalue < ALPHA:
            print("   *Significant difference (p={:.4f})".format(kw.pvalue))
            print("  Post hoc Mann-Whitney U:")
            for nt_comb in itertools.combinations(
                small_df.loc[:, "noise_type"].unique(), 2
            ):
                mwu = stats.mannwhitneyu(
                    *(
                        small_df.loc[:, "threshold"][
                            small_df.loc[:, "noise_type"] == nt
                        ]
                        for nt in nt_comb
                    ),
                    alternative="two-sided",
                )
                if mwu.pvalue < ALPHA:
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
    return


def fit_lme(
    df: pd.DataFrame, weighted: bool = True, print_results: bool = False
) -> tuple([pd.DataFrame, pd.DataFrame, rnlme]):
    """Apply linear mixed model to ABR data

    This function uses an R instance to fit a linear mixed effects (LME) model
    to ABR wave 1 amplitude data. The following formula is used:

        log(wave1_amp) ~ level_db * noise_type

    The interaction term, indicated by "*", allows for different slopes for the
    noise_types. This syntax in R essentially expands to the function

        dependent_variable = independent_variable + factor + independent_variable:factor

    and allows for different slopes for each NoiseType.

    After fitting the LME model the model residuals are tested for
    heteroskedacity using Levene's test (see also
    https://en.wikipedia.org/wiki/Levene's_test). A message will be printed if
    the test fails to show homoskedacity. However, visual inspection is
    generally considered a better method.

    The data can be weighted in the model to counter heteroskedacity. This is
    done by allowing each subject to have a separate intercept. Each intercept
    for a given noise_type will be the mean of the intercepts of the subjects
    within that noise_type. The output of the model gives an estimate of the
    variance around that estimate, which reflects the variability of the
    intercepts fit for each subject. This weighting is an input parameter.

    This function will look at fields id, noise_type, level_db, and wave 1
    amplitude only.

    Args:
        df (pd.DataFrame, optional): Dataframe with data from which predictions
            will be made. It is assumed to have the following column names:
                id : category
                noise_type : category
                level_db : float64
                wave1_amp : float64
            IMPORTANT! Pandas dataframe columns MUST have appropriate datatype
            (i.e., categorical, float64)!
        weighted (bool, optional): Switch to enable weighting of data to
            counter heteroskedactiy. Defaults to True.
        print_results (bool, optional): Switch to enable printing output of
            model results and statistical tests.. Defaults to False.

    Returns:
        tuple([predictions, comparisons, model]:
            predictions (pd.DataFrame): Dataframe with predictions of wave1_amp
                for each combination of noise_type and level_db.
            comparisons (pd.DataFrame): Dataframe with comparisons of
                wave1_amp for all noise_types within each level_db, including
                p-values of their similarity.
            model (rnlme): Instance of the model generated in R.
    """

    # Convert pandas DataFrame to R dataframe
    with localconverter(default_converter + pandas2ri.converter) as cv:
        r_df = robjects.conversion.py2rpy(df)

    # Define model
    formula_model = robjects.Formula("log(wave1_amp) ~ level_db * noise_type")
    formula_weights = robjects.Formula(
        "~1|level_db"
    )  # separate intercept for each subject
    independent_var = "level_db"
    formula_random = robjects.Formula("~1|id")

    # Determine weights
    if weighted:
        weights = rnlme.varIdent(form=formula_weights)
    else:
        weights = robjects.r("NULL")

    # Fit model
    # ML estimates are unbiased for the fixed effects but biased for the random
    # effects, whereas the REML estimates are biased for the fixed effects and
    # unbiased for the random effects.
    model = rnlme.lme(
        formula_model, method="REML", random=formula_random, data=r_df, weights=weights
    )  # REML preferred over ML for small sample sizes

    if print_results:
        print(rstats.anova(model, type="marginal"))

    # Test for heteroskedacity
    levene_model = rcar.leveneTest(
        rstats.residuals(model), group=r_df.rx2("noise_type")
    )
    pval_levene_model = levene_model.rx("Pr(>F)")[0][0]

    if print_results:
        print("")
        print(levene_model)

    if pval_levene_model < 0.05:
        print(
            f"  Residuals could be heteroskedastic (Levene's test, p={pval_levene_model:.4f})."
        )
        print("")
    else:
        print("  Residuals are homoskedastic.")
        print("")

    # Make predictions with model

    # Specify variables to be used as predictors.
    prediction_grid = rbase.expand_grid(
        noise_type=rbase.unique(rbase.c(rbase.as_character(r_df.rx2("noise_type")))),
        level_db=rbase.sort(rbase.unique(rbase.c(r_df.rx2(independent_var)))),
    )

    # Make predictions
    predicted_values = rstats.predict(model, prediction_grid, level=0)

    # Combine predictors with predictions
    predictions = rbase.data_frame(prediction_grid, predicted=predicted_values)

    # Post hoc tests by stimulus intensity, corrected for multiple comparisons
    tt = rlsmeans.lstrends(
        model,
        "noise_type",
        var=independent_var,
        at=rbase.list(
            level_db=rbase.sort(rbase.unique(rbase.c(r_df.rx2(independent_var))))
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
        {"noise_type": "category"}
    )
    comparisons_out = pandas2ri.rpy2py_dataframe(rbase.summary(comparisons))

    return predictions_out, comparisons_out, model
