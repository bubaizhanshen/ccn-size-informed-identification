"""Keep local inputs, generated outputs and temporary files outside Git."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CCN_WORKSPACE", "workspace")).expanduser().resolve()
WORK = ROOT / "cache"
WORK.mkdir(parents=True, exist_ok=True, mode=0o700)
for name in (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
):
    os.environ[name] = str(WORK)
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

# Fixed chronological boundaries used in the manuscript. These are analysis
# configuration, not observations or test-derived optimization parameters.
TEST_BOUNDARIES = {
    "ANX": "2020-03-23 00:00:00+00:00",
    "COR": "2019-02-23 00:00:00+00:00",
    "ENA": "2023-09-01 00:00:00+00:00",
    "GUC": "2023-01-06 05:00:00+00:00",
    "MAO": "2014-12-03 00:00:00+00:00",
    "MOS": "2020-06-27 02:00:00+00:00",
    "SBS_CP": "2011-02-28 00:00:00+00:00",
    "SBS_SPL": "2011-02-27 00:00:00+00:00",
    "SGP": "2021-11-23 00:00:00+00:00",
}
