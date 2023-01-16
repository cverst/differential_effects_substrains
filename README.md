# Description
This repository was used for analysis of the hearing data published in [Hearing Research](https://www.sciencedirect.com/journal/hearing-research) as "[Differential effects of noise exposure between substrains of CBA mice](https://doi.org/10.1016/j.heares.2021.108395)".

The repository includes
1. All ABR wave 1 data, DP data, and histological counts reported in the publication
2. Code for data extraction from Tucker-Davis Technologies' .awf files
3. Code for data preprocessing
4. Code for data visualization and statistical analysis
5. Examples of how to use the programs to perform 2-4

# Setup
Make sure to have both Python and R installed on your machine. A conda installation is recommended to facilitate integration of the two languages.

1. Clone the repository
First clone this repository on your machine and navigate to its directory.

`git clone https://github.com/cverst/differential_effects_substrains`

`cd .\differential_effects_substrains\`

2. Create an environment

Build the environment from the env.yaml file. This makes sure Python 3.7 and R 3.4 are used, and that the right R packages are installed.

`conda env create --file env.yaml`

Then activate the new environment.

`conda activate substrains`

While in the new environment use pip to install python dependencies. Using conda for both the Python and R installation creates conflicts.

`pip install -r requirements.txt`

3. (Optional) Install Jupyter

If Jupyter is preferred over an IDE, install Jupyter.

`pip install jupyter`

`python -m ipykernel install --user --name substrains --display-name "substrains"`

# Data

The [data folder](https://github.com/cverst/differential_effects_substrains/tree/main/data) contains both raw data and the data after preprocessing. The preprocessed data can be recreated from the raw data by running the notebook [Preprocessing_of_data](https://github.com/cverst/differential_effects_substrains/blob/main/Preprocessing_of_data.ipynb). The data is split by substrain and type (ABR or DP).

## ABR

ABR data contains the following fields.

    file_number: a sequential number specifying what data points come from the same .awf file

    substrain: abbreviation of the substrain

    id: subject identifier

    noise_spl: level of the noise the subject was exposed to in dB SPL

    analyzer_id: initials of the person who labeled the data

    experimenter_id: initials of the person who recorded the data

    noise_type: time of day the subject was noise exposed (baseline (not yet exposed), ZT3, ZT15)

    abr_time: experimental time point of recording (baseline, 24h (hours), 2w (weeks))

    level_db: stimulus level of the ABR recording in dB SPL

    freq_hz: stimulus frequency of the ABR recording in Hz

    wave1_amp: amplitude of wave 1 of the ABR recording in nV

    threshold: visually identified ABR threshold in dB SPL

Preprocessed data also includes the following additional fields.

    rlm: predicted log(wave 1 amplitude) based on a robust linear model in nV

    rlm_error: difference between "wave1_amp" and "rlm"

    standardization_std: standard deviation of "rlm_error"

    rlm_error_standardized: "rlm_error" divided by "standardization_std"

    is_outlier: True for any "wave1_amp" that lies outside the 95% confidence interval of the skewed normal distribution fitted to all standardized "wave1_amp" values, False otherwise

    confint_low: lower boundary of the scaled 95% confidence interval of the skewed normal distribution around fitted to all standardized "wave1_amp" values

    confint_high: upper boundary of the scaled 95% confidence interval of the skewed normal distribution around fitted to all standardized "wave1_amp" values

## DP

DP data contains the following fields.

    file_number: a sequential number specifying what data points come from the same .awf file

    substrain: abbreviation of the substrain

    noise_spl: level of the noise the subject was exposed to in dB SPL

    noise_type: time of day the subject was noise exposed (baseline (not yet exposed), ZT3, ZT15)

    abr_time: experimental time point of recording (baseline, 24h (hours), 2w (weeks))

    id: subject identifier

    experimenter_id: initials of the person who recorded the data

    freq_hz: stimulus frequency of the DP recording in Hz

    level_db: stimulus level of the DP recording in dB SPL

    f1: freqency of primary component 1 in Hz

    f2: frequency of primary component 2 in Hz, same as freq_hz

    level_f1: measured sound pressure level of primary component 1 in dB SPL

    level_f2: measured sound pressure level of primary component 2 in dB SPL

    level_distprod: measured sound pressure level of distortion product (2*f1-f2) in dB SPL

Preprocessed data also includes the following additional field.

    is_outlier: True for any data point where 10 - 20 < intensity1 - intensity2 < 10 + 20, False otherwise

### Remarks

For the publication outliers in the DP dataset are identified based on the 80 dB SPL stimulus only. In the dataset here this has been improved by considering all sound intensities for a given file. This has lead to a minor difference between the "is_outlier" label and the published data.

## Histological counts

The histological counts include both ribbon synapse counts and pre/post-synaptic pairing counts. The data have the following fields.

    id: subject identifier

    freq_hz: estimated characteristic frequency of cochlear location

    counts: number of ribbon synapses or synaptic pairings

    substrain: abbreviation of the substrain

    noise_spl: level of the noise the subject was exposed to in dB SPL

    abr_time: experimental time point of recording (24h (hours), 2w (weeks))

# Examples

## Data extraction

For data extraction, see [this notebook](https://github.com/cverst/differential_effects_substrains/blob/main/data_extraction/Data_to_csv.ipynb).

## Data preprocessing

Data preprocessing and outlier detection are explained in [this notebook](https://github.com/cverst/differential_effects_substrains/blob/main/Preprocessing_of_data.ipynb).

## Data visualization and statistical analysis

The publication that accompanies this repository provides analyses that cover the entire dataset. [This notebook](https://github.com/cverst/differential_effects_substrains/blob/main/Visualization_and_statistics.ipynb) shows examples of how to use this repository's functions to perform these analyses. By changing the selection criteria of the dataset that is parsed to the functions, the analysis can be adjusted to show, e.g., a different treatment group or substrain.
