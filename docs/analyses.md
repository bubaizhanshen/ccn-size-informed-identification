# Analysis guide

Run commands from the repository root. Prepare the inputs as described in
[data preparation](data.md). Scripts use fixed chronological test boundaries
from `analysis/common/runtime.py`; training labels must end before the
relevant validation or test boundary.

## Main analyses

For all nine records:

```bash
python -m analysis.preprocessing.archive
python -m analysis.primary.hourly
python -m analysis.primary.environmental
python -m analysis.robustness.blocks
```

Preprocessing writes the mapped spectra and a complete-cohort diagnostic
dataset. Hourly prediction evaluates every lead from 1 to 24 h. Its primary
cohort requires current PNSD and measured future CCN, not current CCN or
future PNSD. The current-CCN control uses the subset where current CCN is
also measured.

The environmental analysis summarizes seasonal and annual performance,
compares spectra at similar total number concentrations, and identifies
high-CCN six-hour periods at SGP. A target block needs at least four valid
CCN hours; high states use fitting-period calendar-month 75th percentiles.
The block checks compare target definitions, current-CCN controls and
equal-budget selection.

For a smaller hourly-only run, add `--site SGP` to the first two commands.
The environmental stage requires hourly outputs from all nine records.

## Cross-site high-CCN identification

After the main analyses:

```bash
python -m analysis.primary.cross_site --site SGP
python -m analysis.primary.cross_site --site ENA
python -m analysis.primary.cross_site --site GUC
python -m analysis.primary.cross_site --site MAO
python -m analysis.robustness.month_selection
```

Prepare each site's aerosol inputs first. Cross-site tests use 1, 3 and 6 h
leads and compare the same held-out blocks for each predictor set. The last
command compares selection across the whole test period with selection
within each observed year-month; it needs outputs from all four sites.

## Model comparison

The model benchmark uses SGP blocks with complete six-hour predictor
histories and a shared training, validation, calibration and test cohort.
Run the preparation and tabular models first, then the neural models, then
score the combined predictions:

```bash
python -m analysis.models.prepare
python -m analysis.models.tabular
python -m analysis.models.neural
python -m analysis.models.score
```

The comparison includes logistic regression, random forest, LightGBM,
XGBoost, LSTM and Transformer. Tabular models compare the latest observation
with a six-hour history; sequence models use the history. Scaling is fitted
on training observations. Model settings and stopping epochs are selected
before the held-out test; probability calibration uses a separate earlier
period. Neural-model dependencies are in `requirements-neural.txt`.

## Size and persistence diagnostics

```bash
python -m analysis.diagnostics.temporal --site SGP
python -m analysis.diagnostics.size_fraction
python -m analysis.diagnostics.proxy_and_size_range --site SGP
python -m analysis.robustness.parameterization --site SGP
```

The temporal checks compare common prediction origins, an alternative
chronological split and seasonal subsets at 1, 3 and 6 h. Size-fraction
diagnostics use the complete cohort, including current CCN and future PNSD,
to examine cutoff choice and information shared with current CCN.

The proxy-persistence comparison first fits PNSD(t) → CCN(t) using training
observations, freezes that model and carries its estimate forward. It
compares this baseline with direct future-target prediction and a
two-stage transition model using training-period out-of-fold proxies.
Ridge models cover 1–24 h; LightGBM covers 1, 3, 6, 12 and 24 h.

The same script compares 15–300 with 15–500 nm total number, 82–300 with
82–500 nm count, and their corresponding fractions on matched observations.
Records lacking valid coverage to 500 nm are reported as unavailable for
that comparison, while the proxy experiment still runs.

Parameterization checks compare the fixed regression penalty and 23/24-bin
fraction representations. For SGP, run model preparation before this check
if the common-cohort comparison is needed.

## Meteorological controls

After the SGP cross-site analysis and optional ARM input preparation:

```bash
python -m analysis.robustness.meteorology
python -m analysis.robustness.meteorology_nonlinear
```

These compare predictors on matched blocks with and without current
meteorology. The nonlinear check uses LightGBM. Neither uses meteorological
observations from the future target block.

## Output folders

All paths below are relative to `CCN_WORKSPACE`, or `workspace/` by default.

| Folder under `results/` | Contents |
| --- | --- |
| `mapped/` | Mapped PNSD arrays and integration checks |
| `complete_cohort/` | Complete-cohort arrays for diagnostics |
| `hourly_leads/` | Hourly predictions, support counts and paired gains |
| `environmental_endpoints/` | Seasonal, annual, physical and SGP block comparisons |
| `block_robustness/` | Matched-block controls and target-definition checks |
| `cross_site/SITE/` | Site-specific block predictions and selection results |
| `model_benchmark/` | Shared cohort, model predictions and comparison scores |
| `temporal_information/` | Common-origin and alternative-split diagnostics |
| `complete_diagnostics/` | Size-fraction and cutoff diagnostics |
| `proxy_and_size_range/SITE/` | Proxy-persistence and 300/500 nm comparisons |
| `proxy_and_size_range/month_relative/` | Global versus within-month selection |
| `model_robustness/` | Penalty and fraction-representation checks |
| `meteorology/`, `meteorology_nonlinear/` | SGP weather-adjusted comparisons |

Tables include sample support and paired block-bootstrap intervals. Local
JSON records contain settings, input filenames and sizes, and processing
checks. They are generated outputs, not repository files.
