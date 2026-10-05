"""Evaluate audited hourly PNSD -> measured CCN links at every 1-24 h lead.

The primary estimand is the held-out, within-lead difference between a total-
number model and the same model plus the 82-300 nm number fraction. A separate
comparison conditions on measured current CCN. Each lead uses its own minimally
eligible origins under the same frozen calendar boundary; common-origin replays
quantify support sensitivity. No temporal smoothing or decay law is fitted.
"""

from __future__ import annotations
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
import argparse
import json
import numpy as np
import pandas as pd
from analysis.diagnostics import temporal as audited
from analysis.diagnostics import size_fraction as previous

OUT = ROOT / "results/hourly_leads"
LEADS = tuple(range(1, 25))
SEED = 270927
MIN_FIT_HOURS = 100
MIN_TEST_HOURS = 100


def _scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2_log": previous.r2(y, pred),
        "mae_log": float(np.mean(np.abs(y - pred))),
        "mae_ccn_cm3": float(np.mean(np.abs(np.expm1(y) - np.expm1(pred)))),
    }


def _gain(
    y: np.ndarray, base: np.ndarray, added: np.ndarray, dates: np.ndarray, seed: int
) -> dict[str, float]:
    delta, lo_day, hi_day = previous.paired_delta(y, base, added, dates, seed)
    lo_7day, hi_7day = audited.block_interval(y, base, added, dates, 7, seed + 10000)
    return {
        "delta_r2": delta,
        "lo95_day": lo_day,
        "hi95_day": hi_day,
        "lo95_7day": lo_7day,
        "hi95_7day": hi_7day,
        "mae_reduction_cm3": _scores(y, base)["mae_ccn_cm3"]
        - _scores(y, added)["mae_ccn_cm3"],
    }


def _fit_masks(
    ts: pd.DatetimeIndex, eligible: np.ndarray, cut: pd.Timestamp, lead: int
) -> tuple[np.ndarray, np.ndarray]:
    fit = eligible & np.asarray(ts + pd.Timedelta(hours=lead) < cut)
    test = eligible & np.asarray(ts >= cut)
    return (fit, test)


def analyze_site(site: str) -> None:
    d = audited.read_site(site, leads=LEADS)
    ts = d["ts"]
    cut = d["cut"]
    current = d["eligible"][0]
    x_total = d["aux"]
    x_size = np.column_stack((x_total, d["f82"]))
    x_current = np.column_stack((x_total, d["y"][0]))
    x_current_size = np.column_stack((x_current, d["f82"]))
    test_calendar = np.asarray(ts >= cut)
    support, scores, gains, predictions = ([], [], [], [])
    primary_preds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lead in LEADS:
        y = d["y"][lead]
        fit, test = _fit_masks(ts, d["eligible"][lead], cut, lead)
        dates = np.asarray(ts[test].date, dtype=str)
        control_fit, control_test = _fit_masks(
            ts, d["eligible"][lead] & current, cut, lead
        )
        support.append(
            {
                "site": site,
                "lead_h": lead,
                "n_current_pnsd": int(d["current_ok"].sum()),
                "n_fit": int(fit.sum()),
                "n_test": int(test.sum()),
                "n_test_dates": int(len(np.unique(dates))),
                "n_test_calendar_current_pnsd": int(
                    (d["current_ok"] & test_calendar).sum()
                ),
                "n_control_fit": int(control_fit.sum()),
                "n_control_test": int(control_test.sum()),
                "test_start": str(ts[test].min()) if test.any() else "",
                "test_end": str(ts[test].max()) if test.any() else "",
                "status": (
                    "supported"
                    if fit.sum() >= MIN_FIT_HOURS and test.sum() >= MIN_TEST_HOURS
                    else "insufficient_hours"
                ),
            }
        )
        if fit.sum() < MIN_FIT_HOURS or test.sum() < MIN_TEST_HOURS:
            continue
        p_total = audited.fit_predict(x_total, y, fit, test)
        p_size = audited.fit_predict(x_size, y, fit, test)
        aligned_total = np.full(len(ts), np.nan)
        aligned_size = np.full(len(ts), np.nan)
        aligned_total[test] = p_total
        aligned_size[test] = p_size
        primary_preds[lead] = (aligned_total, aligned_size)
        for model, pred in (("total", p_total), ("total_plus_f82", p_size)):
            scores.append(
                {
                    "site": site,
                    "lead_h": lead,
                    "cohort": "minimal",
                    "model": model,
                    "n_fit": int(fit.sum()),
                    "n_test": int(test.sum()),
                    "n_test_dates": int(len(np.unique(dates))),
                    **_scores(y[test], pred),
                }
            )
        gains.append(
            {
                "site": site,
                "lead_h": lead,
                "cohort": "minimal",
                "comparison": "total_plus_f82_minus_total",
                "n_fit": int(fit.sum()),
                "n_test": int(test.sum()),
                "n_test_dates": int(len(np.unique(dates))),
                **_gain(y[test], p_total, p_size, dates, SEED + 100 * lead + len(site)),
            }
        )
        row = pd.DataFrame(
            {
                "timestamp": ts[test].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lead_h": lead,
                "target_ccn_cm3": np.expm1(y[test]),
                "pred_total_ccn_cm3": np.expm1(p_total),
                "pred_total_plus_f82_ccn_cm3": np.expm1(p_size),
            }
        )
        row["pred_current_ccn_ccn_cm3"] = np.nan
        row["pred_current_ccn_plus_f82_ccn_cm3"] = np.nan
        if control_fit.sum() >= MIN_FIT_HOURS and control_test.sum() >= MIN_TEST_HOURS:
            pc = audited.fit_predict(x_current, y, control_fit, control_test)
            pcs = audited.fit_predict(x_current_size, y, control_fit, control_test)
            control_dates = np.asarray(ts[control_test].date, dtype=str)
            for model, pred in (
                ("total_plus_current_ccn", pc),
                ("total_plus_current_ccn_plus_f82", pcs),
                ("current_ccn_identity", d["y"][0][control_test]),
            ):
                scores.append(
                    {
                        "site": site,
                        "lead_h": lead,
                        "cohort": "current_ccn_available",
                        "model": model,
                        "n_fit": int(control_fit.sum()),
                        "n_test": int(control_test.sum()),
                        "n_test_dates": int(len(np.unique(control_dates))),
                        **_scores(y[control_test], pred),
                    }
                )
            gains.append(
                {
                    "site": site,
                    "lead_h": lead,
                    "cohort": "current_ccn_available",
                    "comparison": "current_ccn_plus_f82_minus_current_ccn",
                    "n_fit": int(control_fit.sum()),
                    "n_test": int(control_test.sum()),
                    "n_test_dates": int(len(np.unique(control_dates))),
                    **_gain(
                        y[control_test],
                        pc,
                        pcs,
                        control_dates,
                        SEED + 5000 + 100 * lead + len(site),
                    ),
                }
            )
            control_at_primary_rows = np.flatnonzero(control_test[test])
            row.loc[control_at_primary_rows, "pred_current_ccn_ccn_cm3"] = np.expm1(pc)
            row.loc[control_at_primary_rows, "pred_current_ccn_plus_f82_ccn_cm3"] = (
                np.expm1(pcs)
            )
        predictions.append(row)
    for name, common_leads in (
        ("common_1to6", tuple(range(1, 7))),
        ("common_1to24", LEADS),
    ):
        if not all((h in primary_preds for h in common_leads)):
            continue
        common = test_calendar & np.logical_and.reduce(
            [d["eligible"][h] for h in common_leads]
        )
        common_dates = np.asarray(ts[common].date, dtype=str)
        support.append(
            {
                "site": site,
                "lead_h": 0,
                "cohort": name,
                "n_test": int(common.sum()),
                "n_test_dates": int(len(np.unique(common_dates))),
                "status": (
                    "supported"
                    if common.sum() >= MIN_TEST_HOURS
                    else "insufficient_hours"
                ),
            }
        )
        if common.sum() < MIN_TEST_HOURS:
            continue
        for lead in common_leads:
            y = d["y"][lead][common]
            p_total = primary_preds[lead][0][common]
            p_size = primary_preds[lead][1][common]
            if not (np.isfinite(p_total).all() and np.isfinite(p_size).all()):
                raise ValueError(f"{site}: missing predictions on {name} at {lead} h")
            for model, pred in (("total", p_total), ("total_plus_f82", p_size)):
                scores.append(
                    {
                        "site": site,
                        "lead_h": lead,
                        "cohort": name,
                        "model": model,
                        "n_fit": np.nan,
                        "n_test": int(common.sum()),
                        "n_test_dates": int(len(np.unique(common_dates))),
                        **_scores(y, pred),
                    }
                )
            gains.append(
                {
                    "site": site,
                    "lead_h": lead,
                    "cohort": name,
                    "comparison": "total_plus_f82_minus_total",
                    "n_fit": np.nan,
                    "n_test": int(common.sum()),
                    "n_test_dates": int(len(np.unique(common_dates))),
                    **_gain(
                        y,
                        p_total,
                        p_size,
                        common_dates,
                        SEED + 9000 + 100 * lead + len(site),
                    ),
                }
            )
    OUT.mkdir(parents=True, exist_ok=True, mode=448)
    pd.DataFrame(support).to_csv(OUT / f"{site}_support.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / f"{site}_scores.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / f"{site}_gains.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / f"{site}_test_predictions.csv.gz", index=False, compression="gzip"
    )
    manifest = {
        "site": site,
        "leads_h": list(LEADS),
        "source_csv_files": d["source_files"],
        "mapped24_audit_file": d["mapped_audit_file"],
        "mapped_total_over_frozen_q05_median_q95": d["conservation"],
        "preprocessing_gate_passed": True,
        "frozen_test_boundary": str(cut),
        "primary_eligibility": "valid current mapped PNSD and measured future CCN with instrument QC",
        "current_control_eligibility": "primary eligibility plus measured current CCN with instrument QC",
        "model": "median imputation, fitted standardization, Ridge(alpha=10) on log1p(CCN)",
        "source_pnsd_semantics": "dN/dlog10Dp; conservatively mapped to 15-300 nm bin number concentration",
        "support_rule": {
            "min_fit_hours": MIN_FIT_HOURS,
            "min_test_hours": MIN_TEST_HOURS,
        },
        "bootstrap": "1000 date and 7-day nonoverlapping block draws on fixed test predictions",
        "interpretation": "held-out predictive information, not aerosol causation or a decay law",
    }
    (OUT / f"{site}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        site,
        "done; supported leads",
        sum((x["status"] == "supported" for x in support if x.get("lead_h") in LEADS)),
        "test hours at 24 h",
        next((x["n_test"] for x in support if x.get("lead_h") == 24)),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=previous.SITES)
    args = parser.parse_args()
    sites = (args.site,) if args.site else previous.SITES
    for site in sites:
        analyze_site(site)


if __name__ == "__main__":
    main()
