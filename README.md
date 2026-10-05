# CCN size-informed identification

Python code for testing how particle-size information improves short-lead
prediction of cloud condensation nuclei (CCN) and identification of high-CCN
periods. The analyses compare total particle number, size-selected counts and
size fractions using hourly aerosol observations.

## Installation

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the LSTM and Transformer comparison, also install:

```bash
python -m pip install -r requirements-neural.txt
```

## Data and workspace

Obtain the hourly SMPS and CCN files from the
[Figshare archive](https://figshare.com/articles/dataset/27913806) and place
them in `workspace/inputs/figshare_27913806/selected/`. See
[data preparation](docs/data.md) for the folder layout, required columns and
optional ARM meteorological inputs.

Inputs and generated results stay in `workspace/`, which is excluded from
Git. To use another location:

```bash
export CCN_WORKSPACE=/path/to/ccn-workspace
```

## Running the analysis

Run these commands from the repository root:

```bash
python -m analysis.preprocessing.archive
python -m analysis.primary.hourly
python -m analysis.primary.environmental
python -m analysis.robustness.blocks
python -m analysis.primary.cross_site --site SGP
```

The scripts write tables and predictions to `workspace/results/`. To run
only hourly SGP prediction, add `--site SGP` to the first two commands. See the
[analysis guide](docs/analyses.md) for cross-site tests, model comparisons,
seasonal checks and proxy-persistence diagnostics.

## Repository layout

```text
analysis/
  preprocessing/  Archive preparation and diagnostic cohorts
  primary/        Hourly prediction and high-CCN period identification
  models/         Logistic, forest, boosting, LSTM and Transformer comparisons
  diagnostics/    Size fractions, temporal checks and proxy persistence
  robustness/     Block definitions, meteorology and model settings
  common/         Shared readers, size integration and workspace paths
docs/             Data preparation and run instructions
tests/            Small tests using generated inputs
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests do not require downloaded observations. Data, model weights,
manuscripts, figures and plotting scripts are not included.
