"""Prepare a leakage-checked, common-cohort SGP six-hour CCN state dataset."""

from __future__ import annotations

from analysis.common.files import file_info
from pathlib import Path
import os
from analysis.common.runtime import ROOT
from analysis.common.runtime import WORK

os.umask(63)
WORK.mkdir(parents=True, exist_ok=True, mode=448)
for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[key] = str(WORK)
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import json
import numpy as np
import pandas as pd
from analysis.diagnostics import temporal as audited

OUT = ROOT / "results/model_benchmark"
PREVIOUS = ROOT / "results/environmental_endpoints"
PROTOCOL = Path(__file__)


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    previous_manifest = json.loads((PREVIOUS / "manifest.json").read_text())
    d = audited.read_site("SGP", leads=(1, 3, 6))
    if d["mapped_audit_file"] != previous_manifest["source_pnsd_audit_file"]:
        raise ValueError("PNSD preprocessing audit differs from prior endpoint")
    if d["source_files"] != previous_manifest["source_smps_files"]:
        raise ValueError("SMPS source files differ from prior endpoint")
    blocks_path = PREVIOUS / "sgp_six_hour_target_blocks.csv"
    thresholds_path = PREVIOUS / "sgp_fit_month_thresholds.csv"
    blocks = pd.read_csv(
        blocks_path, parse_dates=["block_start", "block_end", "origin_3h"]
    )
    thresholds = pd.read_csv(thresholds_path).set_index("month")
    if set(thresholds.index) != set(range(1, 13)):
        raise ValueError("frozen fit-month thresholds incomplete")
    bstart = pd.DatetimeIndex(blocks.block_start)
    bend = pd.DatetimeIndex(blocks.block_end)
    origin = pd.DatetimeIndex(blocks.origin_3h)
    if blocks.block_start.duplicated().any() or not bstart.is_monotonic_increasing:
        raise ValueError("duplicate or unordered six-hour block starts")
    if not (bend - bstart == pd.Timedelta(hours=6)).all():
        raise ValueError("non-six-hour target blocks")
    if not (bstart - origin == pd.Timedelta(hours=3)).all():
        raise ValueError("non-three-hour prediction origin")
    times = np.column_stack(
        [origin - pd.Timedelta(hours=k) for k in (5, 4, 3, 2, 1, 0)]
    )
    index = d["ts"].get_indexer(times.ravel()).reshape(times.shape)
    safe = np.maximum(index, 0)
    valid = (index >= 0) & d["current_ok"][safe]
    complete_history = valid.all(axis=1)
    eligible = (blocks.n_valid_ccn_hours.to_numpy(int) >= 4) & complete_history
    cal_cut = pd.Timestamp(previous_manifest["calibration_cut"])
    outer = pd.Timestamp(previous_manifest["outer_test_cut"])
    fit = eligible & np.asarray(bend <= cal_cut)
    cal = eligible & np.asarray(bstart >= cal_cut) & np.asarray(bend <= outer)
    test = eligible & np.asarray(origin >= outer)
    if np.any(fit & cal) or np.any(fit & test) or np.any(cal & test):
        raise ValueError("temporal partitions overlap")
    fit_dates = pd.DatetimeIndex(bstart[fit].floor("D").unique()).sort_values()
    inner_cut = fit_dates[int(np.floor(0.8 * len(fit_dates)))]
    train = fit & np.asarray(bend <= inner_cut)
    val = fit & np.asarray(bstart >= inner_cut)
    if np.any(train & val) or np.any(train & cal) or np.any(val & cal):
        raise ValueError("inner train/validation partitions overlap")
    use = train | val | cal | test
    if min(train.sum(), val.sum(), cal.sum(), test.sum()) < 100:
        raise ValueError("insufficient common-cohort partition support")
    month = bstart.month.to_numpy()
    high_threshold = thresholds.loc[month, "high_threshold_cm3"].to_numpy(float)
    y_all = (blocks.mean_ccn_cm3.to_numpy(float) >= high_threshold).astype("int8")
    if min((y_all[m].sum() for m in (train, val, cal, test))) < 20:
        raise ValueError("insufficient high-state events in a partition")
    old = pd.read_csv(
        PREVIOUS / "sgp_six_hour_state_test_predictions.csv.gz",
        parse_dates=["block_start"],
    )
    old = old.loc[(old.min_hours == 4) & (old.lead_h == 3) & (old["tail"] == "high")]
    old_y = old.set_index("block_start").event
    test_times = bstart[test]
    if not test_times.isin(old_y.index).all() or not np.array_equal(
        y_all[test], old_y.loc[test_times].to_numpy(int)
    ):
        raise ValueError("common-cohort test labels differ from previous endpoint")
    selected = np.flatnonzero(use)
    hist = index[selected]
    if np.any(hist < 0):
        raise ValueError("selected history contains an absent PNSD hour")
    totals = d["total"][hist].astype("float32")
    f82 = d["f82"][hist].astype("float32")
    bins = d["bins"][hist].astype("float32")
    if not (
        np.isfinite(totals).all()
        and np.isfinite(f82).all()
        and np.isfinite(bins).all()
        and (totals > 0).all()
        and (bins >= 0).all()
    ):
        raise ValueError("invalid audited PNSD values in common cohort")
    if np.max(np.abs(bins.sum(axis=2) / totals - 1)) > 0.0001:
        raise ValueError("PNSD bin count does not close to total")
    fractions = bins / totals[:, :, None]
    n82 = totals * f82
    if not np.allclose(d["n82"][hist], n82, rtol=1e-05, atol=0.0001):
        raise ValueError("N82 differs from audited bins")
    if not np.all(np.asarray(times[selected, -1] < bstart[selected])):
        raise ValueError("predictor leaks into target block")
    start = bstart[selected]
    hour = start.hour.to_numpy(float)
    doy = start.dayofyear.to_numpy(float)
    calendar = np.column_stack(
        (
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        )
    ).astype("float32")
    local = {
        name: mask[selected]
        for name, mask in (("train", train), ("val", val), ("cal", cal), ("test", test))
    }
    OUT.mkdir(parents=True, exist_ok=True, mode=448)
    target = OUT / "common_cohort.npz"
    np.savez_compressed(
        target,
        x_N=np.log1p(totals),
        x_N82=np.log1p(n82),
        x_f82=f82,
        x_fraction24=fractions,
        calendar=calendar,
        y=y_all[selected],
        block_start=start.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(dtype="U20"),
        origin=origin[selected].strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(dtype="U20"),
        target_ccn_cm3=blocks.mean_ccn_cm3.to_numpy(float)[selected],
        train=local["train"],
        val=local["val"],
        cal=local["cal"],
        test=local["test"],
    )
    manifest = {
        "analysis_code_file": file_info(PROTOCOL),
        "prior_endpoint_manifest_file": file_info(PREVIOUS / "manifest.json"),
        "prior_blocks_file": file_info(blocks_path),
        "prior_thresholds_file": file_info(thresholds_path),
        "pnsd_audit_file": d["mapped_audit_file"],
        "smps_source_files": d["source_files"],
        "cohort_file": file_info(target),
        "train_val_cut": str(inner_cut),
        "cal_cut": str(cal_cut),
        "outer_test_cut": str(outer),
        "feature_times": "exact hourly PNSD origin-5 h through origin; no target-block PNSD",
        "pnsd_semantics": "audited 24-bin particle number concentration per bin, 15-300 nm",
        "N82_semantics": "audited bins integrated above 82 nm with partial log-bin overlap; not independent instrument",
        "target": "0.4% SS column B six-hour mean high state; >=4/6 valid CCN hours; frozen fit-month p75",
        "partitions": {
            k: {
                "n": int(v.sum()),
                "events": int(y_all[selected][v].sum()),
                "days": int(start[v].floor("D").nunique()),
            }
            for k, v in local.items()
        },
        "previous_test_labels_reproduced": True,
        "opened_archive_not_blind": True,
    }
    (OUT / "common_cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "partitions": manifest["partitions"],
                "cohort_file": manifest["cohort_file"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
