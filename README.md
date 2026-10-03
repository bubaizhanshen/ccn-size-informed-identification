# Size-informed identification of high-CCN periods

Analysis code for **Particle-Size Information Improves Short-Lead Identification of High-CCN Periods**.

This repository contains necessary analysis code and run instructions only.
It does not contain observations, derived datasets, model weights, manuscripts,
figures, plotting code, download scripts, or account credentials.

## Setup and external inputs

Use Python 3.11 or later. Install `requirements.txt`; install
`requirements-neural.txt` only for the LSTM/Transformer comparison.

Obtain the harmonized hourly observations separately from the
[Figshare archive](https://figshare.com/articles/dataset/27913806).
Retain original filenames and columns; place the files under:

```
workspace/inputs/figshare_27913806/selected/
  smps/       # *_smps_hour*.csv
  ccn/        # *_ccn_colb_hour*.csv
  ccn_cola/   # *_ccn_cola.csv (SBS records)
```

For the optional SGP meteorological controls, obtain daily `sgpmetE13.b1`
NetCDF files from the [ARM Data Center](https://doi.org/10.5439/1786358)
for 2017-04-01 through 2023-10-16 and put them in
`workspace/inputs/arm_sgpmetE13_b1/`. Authentication/downloads are handled
outside this repository. Do not put access tokens in analysis scripts.

`CCN_WORKSPACE` may point to a different local working directory. Inputs,
generated numerical outputs and temporary files stay in that directory.
Nothing in it should be committed.

## Run order

Run commands from the repository root. `--site` accepts ANX, COR, ENA, GUC,
MAO, MOS, SBS_CP, SBS_SPL or SGP where applicable.

1. Prepare audited spectra and the diagnostic complete cohort:
   `python analysis/preprocessing_archive.py`
2. Run the primary hourly lead curve and corresponding controls:
   `python analysis/primary_hourly.py`
3. Run SGP environmental endpoints and seasonal/physical diagnostics:
   `python analysis/primary_environmental.py`
4. Run block-target robustness:
   `python analysis/robustness_blocks.py`
5. Run cross-site high-CCN identification (repeat for the desired site):
   `python analysis/primary_cross_site.py --site SGP`
   (ENA, GUC and MAO provide the additional supported records.)
6. Run complete-cohort size diagnostics:
   `python analysis/diagnostic_size_fraction.py`
7. Run concurrent-proxy/persistence and 300-versus-500 nm tests:
   `python analysis/diagnostic_proxy_and_size_range.py --site SGP`
   Repeat for the other records. The code explicitly records unavailable
   15–500 nm coverage instead of imputing unmeasured large-particle channels.

## Additional checks

| Purpose | Analysis modules | Prerequisite |
| --- | --- | --- |
| Seasonality, common origins, chronological split | `shared_temporal.py` (executable), `primary_hourly.py`, `primary_environmental.py` | Prepared archive |
| Fixed penalty and 23/24-fraction parameterizations | `robustness_parameterization.py --site SITE` | Size diagnostics; for SGP, prepared model cohort |
| Comparable tabular and neural models | `model_prepare.py`, `model_tabular.py`, `model_neural.py`, `model_score.py` (in that order) | SGP environmental endpoints; TensorFlow only for neural models |
| Meteorological/QC controls | `robustness_meteorology.py`, `robustness_meteorology_nonlinear.py` | SGP cross-site outputs and external ARM meteorology |
| Global versus within-year-month selection | `robustness_month_selection.py` | Cross-site outputs for SGP, ENA, GUC and MAO |

Tabular families are logistic regression, random forest, LightGBM and
XGBoost; neural families are LSTM and Transformer with the manuscript's
fixed seeds, architecture, temporal partitions and calibration procedure.

## Analysis conventions

- The primary hourly cohort requires valid earlier PNSD and later measured
  CCN. Concurrent CCN and later PNSD are not primary eligibility conditions.
  The complete cohort is used only for named diagnostic comparisons.
- CCN is evaluated at 0.4% supersaturation; concentrations are in cm−3.
  Spectral `D_*` columns are dN/dlog10(Dp), not per-bin counts. The preprocessing
  audit integrates/remaps density to 24 equal-log bins spanning 15–300 nm,
  checks against the independent archive total, and rejects failed gates.
- N82 and f82 use fractional overlap with the 82 nm cutoff. They are derived
  from the spectrum, not separate instruments. No prediction label is used
  to choose that literature-specified cutoff.
- Holdouts are chronological with label-end purging. Thresholds, scalers,
  imputers and calibration maps use pretest periods only.
- The concurrent proxy is trained on PNSD(t) → CCN(t), then frozen and carried
  forward unchanged. Direct future-target and matched two-stage alternatives
  use their stated training populations.
- Six-hour targets require at least four valid CCN hours. High states use
  fitting-period calendar-month 75th percentiles. Top-decile selection uses
  stable ranks; paired calendar-block resampling keeps predictions fixed.
- Tests use the opened archive, not an independently blinded dataset.
  Observational conditional associations are not causal interventions.

Scripts write numerical tables, predictions and provenance manifests locally;
they do not produce figures. Completed-output guards prevent accidental
overwriting. Use a fresh workspace for a new run.

## Basic checks

`python -m unittest discover -s tests -v` checks spectral cutoff integration,
grid-resolution invariance, label-end purging, stable selection and paired
resampling using arrays generated in memory. No test dataset is included.
