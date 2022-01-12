# Description
This repository was used for analysis of the ABR data published as "Differential effects of noise exposure between substrains of CBA mice." This article has been accepted to [Hearing Research](https://www.sciencedirect.com/journal/hearing-research).

The repository includes
1. All ABR wave 1 data and DPOAE data reported in the publication
2. Code for data extraction from Tucker-Davis Technologies' .awf files
3. Code for data preprocessing
4. Code for data visualization and statistical analysis
5. Examples of how to use the programs to perform 2-4

# Setup
Make sure to have both Python and R installed on your machine. A conda installation is recommended to facilitate integration of the two languages.

1. Clone the repository
First clne this repository on your machine and navigate to its directory.
`git clone https://github.com/cverst/differential_effects_substrains`
`cd .\differential_effects_substrains\`
2. Create an environment
Build the environment from the env.yaml file. This makes sure Python 3.7 and R 3.4 are used, and that the right R packages are installed.
`conda create --file env.yaml`
Then activate the new environment.
`conda activate substrains`
While in the new environment use pip to install python dependencies. Using conda for both the Python and R installation creates conflicts.
`pip install -r requirements.txt`
If Jupyter is preferred over an IDE, install Jupyter.
`pip install jupyter`
`python -m ipykernel install --user --name substrains --display-name "substrains"`

# Examples

## Data extraction
For data extraction, see [this notebook](https://github.com/cverst/differential_effects_substrains/blob/main/data_extraction/Data_to_csv.ipynb).

## Data preprocessing
Data preprocessing and outlier detection are explained in [this notebook](https://github.com/cverst/differential_effects_substrains/blob/main/Preprocessing_of_data.ipynb)

## Data visualization and statistical analysis
The publication that accompanies this repository provides analyses that cover the entire dataset. [This notebook](https://github.com/cverst/differential_effects_substrains/blob/main/Visualization_and_statistics.ipynb) shows examples of how to use this repository's functions to perform these analyses. By changing the selection criteria of the dataset that is parsed to the functions, the analysis can be adjusted to show, e.g., a different treatment group or substrain.
