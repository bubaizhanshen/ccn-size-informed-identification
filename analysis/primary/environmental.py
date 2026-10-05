"""Seasonal support and measured six-hour CCN states on audited PNSD.

These are opened-archive follow-ups, not independent blind validation.
"""

from __future__ import annotations

from analysis.common.files import file_info
from pathlib import Path
import os
import math
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from analysis.preprocessing import complete_cohort as builder
from analysis.diagnostics import temporal as audited
from analysis.diagnostics import size_fraction as spectra

OUT = ROOT / "results/environmental_endpoints"
PRED = ROOT / "results/hourly_leads"
PROTOCOL = Path(__file__)
SITES = spectra.SITES
LEADS = (1, 3, 6, 12, 24)
BLOCK_LEADS = (1, 3, 6)
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEED = 270928


def season_of(month: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(SEASONS, dtype=object)[np.asarray(month, dtype=int) % 12 // 3]


def r2(y: np.ndarray, p: np.ndarray) -> float:
    den = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / den) if den > 0 else np.nan


def paired_error_intervals(
    y: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    timestamps: pd.DatetimeIndex,
    seed: int,
    draws: int = 1000,
) -> dict[str, float]:
    """Paired fixed-prediction bootstrap of seven-day calendar blocks."""
    day = timestamps.asi8 // 86400000000000
    week = day // 7
    _, ix = np.unique(week, return_inverse=True)
    n = np.bincount(ix)
    sy = np.bincount(ix, weights=y)
    sy2 = np.bincount(ix, weights=y * y)
    sse0 = np.bincount(ix, weights=(y - p0) ** 2)
    sse1 = np.bincount(ix, weights=(y - p1) ** 2)
    c = np.expm1(y)
    mae_gain = np.bincount(
        ix, weights=np.abs(c - np.expm1(p0)) - np.abs(c - np.expm1(p1))
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(n), size=(draws, len(n)))
    counts = n[sampled].sum(axis=1)
    sst = sy2[sampled].sum(axis=1) - sy[sampled].sum(axis=1) ** 2 / counts
    delta_r2 = (sse0[sampled].sum(axis=1) - sse1[sampled].sum(axis=1)) / sst
    delta_mae = mae_gain[sampled].sum(axis=1) / counts
    delta_r2 = delta_r2[np.isfinite(delta_r2)]
    return {
        "n_seven_day_blocks": int(len(n)),
        "delta_r2_lo95": (
            float(np.quantile(delta_r2, 0.025)) if len(delta_r2) else np.nan
        ),
        "delta_r2_hi95": (
            float(np.quantile(delta_r2, 0.975)) if len(delta_r2) else np.nan
        ),
        "mae_gain_cm3_lo95": float(np.quantile(delta_mae, 0.025)),
        "mae_gain_cm3_hi95": float(np.quantile(delta_mae, 0.975)),
    }


def seasonal_replay() -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict] = []
    sources: dict[str, str] = {}
    for site in SITES:
        path = PRED / f"{site}_test_predictions.csv.gz"
        manifest = json.loads((PRED / f"{site}_manifest.json").read_text())
        if (
            not manifest.get("preprocessing_gate_passed")
            or manifest.get("frozen_test_boundary") is None
        ):
            raise ValueError(
                f"{site}: hourly prediction preprocessing provenance failed"
            )
        sources[site] = file_info(path)
        df = pd.read_csv(path)
        df = df.loc[df.lead_h.isin(LEADS)].copy()
        ts = pd.DatetimeIndex(pd.to_datetime(df.timestamp, utc=True))
        if df.duplicated(["timestamp", "lead_h"]).any():
            raise ValueError(f"{site}: duplicate hourly prediction pairs")
        df["season"] = season_of(ts.month)
        df["month"] = ts.month
        for lead in LEADS:
            for season in SEASONS:
                x = df.loc[(df.lead_h == lead) & (df.season == season)]
                if x.empty:
                    rows.append(
                        {
                            "site": site,
                            "lead_h": lead,
                            "season": season,
                            "n_hours": 0,
                            "n_dates": 0,
                            "supported": False,
                        }
                    )
                    continue
                when = pd.DatetimeIndex(pd.to_datetime(x.timestamp, utc=True))
                y = np.log1p(x.target_ccn_cm3.to_numpy(float))
                p0 = np.log1p(x.pred_total_ccn_cm3.to_numpy(float))
                p1 = np.log1p(x.pred_total_plus_f82_ccn_cm3.to_numpy(float))
                if not all((np.isfinite(v).all() for v in (y, p0, p1))):
                    raise ValueError(f"{site}: nonfinite seasonal target or prediction")
                n_days = int(len(np.unique(when.floor("D"))))
                data = {
                    "site": site,
                    "lead_h": lead,
                    "season": season,
                    "n_hours": int(len(x)),
                    "n_dates": n_days,
                    "n_test_calendar_months": int(len(set(zip(when.year, when.month)))),
                    "supported": bool(len(x) >= 100 and n_days >= 30),
                    "test_start": str(when.min()),
                    "test_end": str(when.max()),
                    "observed_ccn_median_cm3": float(np.median(np.expm1(y))),
                    "r2_total_log": r2(y, p0),
                    "r2_size_log": r2(y, p1),
                    "delta_r2": r2(y, p1) - r2(y, p0),
                    "mae_total_cm3": float(np.mean(np.abs(np.expm1(y) - np.expm1(p0)))),
                    "mae_size_cm3": float(np.mean(np.abs(np.expm1(y) - np.expm1(p1)))),
                    "mae_gain_cm3": float(
                        np.mean(
                            np.abs(np.expm1(y) - np.expm1(p0))
                            - np.abs(np.expm1(y) - np.expm1(p1))
                        )
                    ),
                }
                if data["supported"]:
                    data.update(
                        paired_error_intervals(
                            y,
                            p0,
                            p1,
                            when,
                            SEED + 100 * lead + len(site) + SEASONS.index(season),
                        )
                    )
                rows.append(data)
        print(site, "seasonal replay complete", flush=True)
    return (pd.DataFrame(rows), sources)


def sgp_blocks(d: dict) -> tuple[pd.DataFrame, str]:
    archive = builder.load_ccn_instrument_only()["SGP"]
    target = pd.to_numeric(archive.N_CCN_mean_STP_B, errors="coerce")
    good = np.isfinite(target) & (target >= 0) & archive.ccn_instrument_ok.astype(bool)
    hourly = pd.Series(
        target.loc[good].to_numpy(float),
        index=pd.DatetimeIndex(archive.loc[good, "timestamp"]),
    )
    if hourly.index.has_duplicates:
        raise ValueError("SGP CCN source has duplicate timestamps")
    starts = pd.date_range(
        hourly.index.min().floor("6h"),
        hourly.index.max().floor("6h"),
        freq="6h",
        tz="UTC",
    )
    values = np.column_stack(
        [
            hourly.reindex(starts + pd.Timedelta(hours=k)).to_numpy(float)
            for k in range(6)
        ]
    )
    count = np.isfinite(values).sum(axis=1)
    mean = np.divide(
        np.nansum(values, axis=1),
        count,
        out=np.full(len(starts), np.nan),
        where=count > 0,
    )
    blocks = pd.DataFrame(
        {
            "block_start": starts,
            "block_end": starts + pd.Timedelta(hours=6),
            "n_valid_ccn_hours": count,
            "mean_ccn_cm3": mean,
            "month": starts.month,
            "season": season_of(starts.month),
        }
    )
    for lead in BLOCK_LEADS:
        origin = starts - pd.Timedelta(hours=lead)
        row = d["ts"].get_indexer(origin)
        valid = row >= 0
        valid[valid] &= d["current_ok"][row[valid]]
        number = np.full(len(starts), np.nan)
        fraction = np.full(len(starts), np.nan)
        number[valid] = d["total"][row[valid]]
        fraction[valid] = d["f82"][row[valid]]
        blocks[f"origin_{lead}h"] = origin
        blocks[f"origin_valid_{lead}h"] = valid
        blocks[f"N15_300_{lead}h"] = number
        blocks[f"f82_{lead}h"] = fraction
    return (
        blocks,
        file_info(
            ROOT / "inputs/figshare_27913806/selected/ccn/SGP_ccn_colb_hour_2017.csv"
        ),
    )


def classifier() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
        ]
    )


def block_features(frame: pd.DataFrame, lead: int) -> dict[str, np.ndarray]:
    start = pd.DatetimeIndex(frame.block_start)
    hour = start.hour.to_numpy(float)
    doy = start.dayofyear.to_numpy(float)
    calendar = np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]
    )
    total = np.log1p(frame[f"N15_300_{lead}h"].to_numpy(float))[:, None]
    fraction = frame[f"f82_{lead}h"].to_numpy(float)[:, None]
    n82 = np.log1p(
        frame[f"N15_300_{lead}h"].to_numpy(float)
        * frame[f"f82_{lead}h"].to_numpy(float)
    )[:, None]
    return {
        "calendar": calendar,
        "calendar_total": np.column_stack((calendar, total)),
        "calendar_total_f82": np.column_stack((calendar, total, fraction)),
        "calendar_N82": np.column_stack((calendar, n82)),
    }


def brier_interval(
    y: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    dates: pd.DatetimeIndex,
    seed: int,
    draws: int = 2000,
) -> tuple[float, float]:
    week = dates.asi8 // 86400000000000 // 7
    _, ix = np.unique(week, return_inverse=True)
    n = np.bincount(ix)
    diff = np.bincount(ix, weights=(y - p0) ** 2 - (y - p1) ** 2)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(n), size=(draws, len(n)))
    vals = diff[sampled].sum(axis=1) / n[sampled].sum(axis=1)
    return tuple((float(q) for q in np.quantile(vals, [0.025, 0.975])))


def six_hour_state(
    d: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    blocks, _ = sgp_blocks(d)
    outer = d["cut"]
    common_pre = (blocks.n_valid_ccn_hours >= 4) & (blocks.block_end <= outer)
    for lead in BLOCK_LEADS:
        common_pre &= blocks[f"origin_valid_{lead}h"]
    dates = pd.DatetimeIndex(
        blocks.loc[common_pre, "block_start"].dt.floor("D").unique()
    ).sort_values()
    if len(dates) < 180:
        raise ValueError("insufficient SGP pretest days for calendar-month threshold")
    cal_cut = dates[int(np.floor(0.8 * len(dates)))]
    common_fit = common_pre & (blocks.block_end <= cal_cut)
    month_support = (
        blocks.loc[common_fit]
        .groupby("month")
        .agg(
            n_fit_blocks=("mean_ccn_cm3", "size"),
            n_fit_days=("block_start", lambda x: x.dt.floor("D").nunique()),
            low_threshold_cm3=("mean_ccn_cm3", lambda x: x.quantile(0.25)),
            high_threshold_cm3=("mean_ccn_cm3", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    if len(month_support) != 12 or (month_support.n_fit_blocks < 30).any():
        raise ValueError("calendar-month thresholds do not have training support")
    low = blocks.month.map(month_support.set_index("month").low_threshold_cm3).to_numpy(
        float
    )
    high = blocks.month.map(
        month_support.set_index("month").high_threshold_cm3
    ).to_numpy(float)
    blocks["fit_month_low_threshold_cm3"] = low
    blocks["fit_month_high_threshold_cm3"] = high
    scores, gains, predictions, seasons = ([], [], [], [])
    for min_hours in (4, 5):
        for lead in BLOCK_LEADS:
            eligible = (blocks.n_valid_ccn_hours >= min_hours) & blocks[
                f"origin_valid_{lead}h"
            ]
            fit = eligible & (blocks.block_end <= cal_cut)
            cal = (
                eligible & (blocks.block_start >= cal_cut) & (blocks.block_end <= outer)
            )
            test = eligible & (blocks[f"origin_{lead}h"] >= outer)
            if (fit & cal).any() or (fit & test).any() or (cal & test).any():
                raise ValueError("SGP six-hour temporal split overlap")
            if min(int(fit.sum()), int(cal.sum()), int(test.sum())) < 100:
                raise ValueError(
                    f"SGP six-hour population too small: min_hours={min_hours!r}, lead={lead!r}"
                )
            features = block_features(blocks, lead)
            for tail in ("high", "low"):
                labels = (
                    blocks.mean_ccn_cm3.to_numpy(float) >= high
                    if tail == "high"
                    else blocks.mean_ccn_cm3.to_numpy(float) <= low
                )
                if any(
                    (
                        labels[mask].sum() < 15 or (~labels[mask]).sum() < 15
                        for mask in (fit, cal, test)
                    )
                ):
                    raise ValueError(
                        f"SGP six-hour class support failed: min_hours={min_hours!r}, lead={lead!r}, tail={tail!r}"
                    )
                pred = blocks.loc[
                    test,
                    [
                        "block_start",
                        "block_end",
                        "season",
                        "month",
                        "n_valid_ccn_hours",
                        "mean_ccn_cm3",
                        f"origin_{lead}h",
                        f"N15_300_{lead}h",
                        f"f82_{lead}h",
                    ],
                ].copy()
                pred = pred.rename(
                    columns={
                        f"origin_{lead}h": "origin",
                        f"N15_300_{lead}h": "N15_300",
                        f"f82_{lead}h": "f82",
                    }
                )
                pred["event"] = labels[test].astype(int)
                pred["tail"] = tail
                pred["lead_h"] = lead
                pred["min_hours"] = min_hours
                pred["fit_month_threshold_cm3"] = (high if tail == "high" else low)[
                    test
                ]
                for name, x in features.items():
                    model = classifier().fit(x[fit], labels[fit].astype(int))
                    pcal = model.predict_proba(x[cal])[:, 1]
                    ptest = model.predict_proba(x[test])[:, 1]
                    cutoff = float(np.quantile(pcal[~labels[cal]], 0.9))
                    alarm = ptest > cutoff
                    pred[f"p_{name}"] = ptest
                    pred[f"alarm_{name}"] = alarm.astype(int)
                    y = labels[test]
                    prior = np.full(
                        int(test.sum()), float(y.size and labels[fit].mean())
                    )
                    scores.append(
                        {
                            "min_hours": min_hours,
                            "lead_h": lead,
                            "tail": tail,
                            "model": name,
                            "n_fit": int(fit.sum()),
                            "n_cal": int(cal.sum()),
                            "n_test": int(test.sum()),
                            "event_fit": int(labels[fit].sum()),
                            "event_cal": int(labels[cal].sum()),
                            "event_test": int(y.sum()),
                            "test_event_rate": float(y.mean()),
                            "mean_test_ccn_cm3": float(
                                blocks.loc[test, "mean_ccn_cm3"].mean()
                            ),
                            "fit_prior_brier": float(brier_score_loss(y, prior)),
                            "brier": float(brier_score_loss(y, ptest)),
                            "average_precision": float(
                                average_precision_score(y, ptest)
                            ),
                            "alarm_cut_cal_neg_p90": cutoff,
                            "hits": int((alarm & y).sum()),
                            "false_alarms": int((alarm & ~y).sum()),
                            "misses": int((~alarm & y).sum()),
                            "false_alarm_rate": float((alarm & ~y).sum() / (~y).sum()),
                        }
                    )
                for base in ("calendar", "calendar_total", "calendar_N82"):
                    y = pred.event.to_numpy(int)
                    p0 = pred[f"p_{base}"].to_numpy(float)
                    p1 = pred.p_calendar_total_f82.to_numpy(float)
                    gain = float(np.mean((y - p0) ** 2 - (y - p1) ** 2))
                    ci = brier_interval(
                        y,
                        p0,
                        p1,
                        pd.DatetimeIndex(pred.block_start),
                        SEED + 100 * lead + 10 * min_hours + len(tail) + len(base),
                    )
                    gains.append(
                        {
                            "min_hours": min_hours,
                            "lead_h": lead,
                            "tail": tail,
                            "base": base,
                            "added": "calendar_total_f82",
                            "n_test": int(len(y)),
                            "event_test": int(y.sum()),
                            "brier_gain": gain,
                            "lo95_7day": ci[0],
                            "hi95_7day": ci[1],
                            "ap_gain": float(
                                average_precision_score(y, p1)
                                - average_precision_score(y, p0)
                            ),
                        }
                    )
                for season in SEASONS:
                    s = pred.loc[pred.season == season]
                    y = s.event.to_numpy(int)
                    if not len(y):
                        continue
                    p0 = s.p_calendar_total.to_numpy(float)
                    p1 = s.p_calendar_total_f82.to_numpy(float)
                    seasons.append(
                        {
                            "min_hours": min_hours,
                            "lead_h": lead,
                            "tail": tail,
                            "season": season,
                            "n_test": int(len(y)),
                            "n_test_days": int(
                                pd.DatetimeIndex(s.block_start).floor("D").nunique()
                            ),
                            "events": int(y.sum()),
                            "brier_total": float(brier_score_loss(y, p0)),
                            "brier_size": float(brier_score_loss(y, p1)),
                            "brier_gain": float(np.mean((y - p0) ** 2 - (y - p1) ** 2)),
                            "ap_total": (
                                float(average_precision_score(y, p0))
                                if y.sum()
                                else np.nan
                            ),
                            "ap_size": (
                                float(average_precision_score(y, p1))
                                if y.sum()
                                else np.nan
                            ),
                        }
                    )
                predictions.append(pred)
    manifest = {
        "outer_test_cut": str(outer),
        "calibration_cut": str(cal_cut),
        "monthly_threshold_fit_common_4of6": True,
        "source_pnsd_audit_file": d["mapped_audit_file"],
        "source_smps_files": d["source_files"],
        "target_source_files": {
            path.name: file_info(path)
            for path in sorted(
                (ROOT / "inputs/figshare_27913806/selected/ccn").glob(
                    "SGP_ccn_colb_hour_*.csv"
                )
            )
        },
        "source_ccn_column": "N_CCN_mean_STP_B; SS_setpoint_B=0.4; instrument QC only",
        "eligible_ccn_hours": int(len(builder.load_ccn_instrument_only()["SGP"])),
        "block_target": "mean of >=4 of 6 valid CCN hours; 5 of 6 sensitivity",
        "monthly_thresholds": "fit-only 25th/75th percentile within UTC calendar month",
        "prediction_inputs": "known block calendar, PNSD integral total, optional f82 at origin",
        "test_state": "opened archive, not blinded or real instrument outage",
    }
    return (
        blocks,
        pd.DataFrame(scores),
        pd.DataFrame(gains),
        pd.concat(predictions, ignore_index=True),
        {
            "manifest": manifest,
            "monthly_threshold_support": month_support,
            "seasonal_state_scores": pd.DataFrame(seasons),
        },
    )


def physical_group_interval(
    frame: pd.DataFrame, seed: int, draws: int = 1000
) -> tuple[float, float]:
    weeks = pd.DatetimeIndex(frame.timestamp).asi8 // 86400000000000 // 7
    unique = np.unique(weeks)
    positions = {week: np.flatnonzero(weeks == week) for week in unique}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([positions[week] for week in sampled])
        selected = frame.iloc[ix]
        low = selected.loc[selected.group == "low", "future_ccn_cm3"]
        high = selected.loc[selected.group == "high", "future_ccn_cm3"]
        if len(low) and len(high):
            differences.append(float(high.median() - low.median()))
    return tuple((float(x) for x in np.quantile(differences, [0.025, 0.975])))


def physical_contrast(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = d["ts"]
    fit = d["eligible"][3] & np.asarray(ts + pd.Timedelta(hours=3) < d["cut"])
    test = d["eligible"][3] & np.asarray(ts >= d["cut"])
    seasons = season_of(ts.month)
    rows, profiles = ([], [])
    z = np.log10(d["diameter"].astype(float))
    edges = np.r_[
        z[0] - (z[1] - z[0]) / 2, (z[:-1] + z[1:]) / 2, z[-1] + (z[-1] - z[-2]) / 2
    ]
    widths = np.diff(edges)
    for season in SEASONS:
        train = fit & (seasons == season)
        evaluate = test & (seasons == season)
        n_lo, n_hi = np.quantile(d["total"][train], [0.3, 0.7])
        train_middle = train & (d["total"] >= n_lo) & (d["total"] <= n_hi)
        f_lo, f_hi = np.quantile(d["f82"][train_middle], [0.25, 0.75])
        test_middle = evaluate & (d["total"] >= n_lo) & (d["total"] <= n_hi)
        low = test_middle & (d["f82"] <= f_lo)
        high = test_middle & (d["f82"] >= f_hi)
        if min(low.sum(), high.sum()) < 20:
            rows.append(
                {
                    "season": season,
                    "status": "insufficient_test_groups",
                    "n_fit_middle": int(train_middle.sum()),
                    "n_test_middle": int(test_middle.sum()),
                    "n_test_low": int(low.sum()),
                    "n_test_high": int(high.sum()),
                }
            )
            continue
        ylo = np.expm1(d["y"][3][low])
        yhi = np.expm1(d["y"][3][high])
        selection = low | high
        comparison = pd.DataFrame(
            {
                "timestamp": ts[selection],
                "group": np.where(high[selection], "high", "low"),
                "future_ccn_cm3": np.expm1(d["y"][3][selection]),
            }
        )
        ci = physical_group_interval(comparison, SEED + SEASONS.index(season))
        rows.append(
            {
                "season": season,
                "status": "descriptive_holdout_contrast",
                "n_fit_middle": int(train_middle.sum()),
                "n_test_middle": int(test_middle.sum()),
                "n_test_low": int(low.sum()),
                "n_test_high": int(high.sum()),
                "fit_N_q30_cm3": float(n_lo),
                "fit_N_q70_cm3": float(n_hi),
                "fit_f82_q25_in_middle": float(f_lo),
                "fit_f82_q75_in_middle": float(f_hi),
                "N_low_median_cm3": float(np.median(d["total"][low])),
                "N_high_median_cm3": float(np.median(d["total"][high])),
                "f82_low_median": float(np.median(d["f82"][low])),
                "f82_high_median": float(np.median(d["f82"][high])),
                "future_CCN_low_median_cm3": float(np.median(ylo)),
                "future_CCN_high_median_cm3": float(np.median(yhi)),
                "future_CCN_median_ratio_high_low": float(
                    np.median(yhi) / np.median(ylo)
                ),
                "future_CCN_median_difference_cm3": float(
                    np.median(yhi) - np.median(ylo)
                ),
                "future_CCN_difference_lo95_7day": ci[0],
                "future_CCN_difference_hi95_7day": ci[1],
            }
        )
        for group, mask in (("low", low), ("high", high)):
            normalized_density = d["bins"][mask] / d["total"][mask, None] / widths
            for j, diameter in enumerate(d["diameter"]):
                profiles.append(
                    {
                        "season": season,
                        "group": group,
                        "diameter_nm": float(diameter),
                        "n_test": int(mask.sum()),
                        "median_normalized_dndlog10dp": float(
                            np.median(normalized_density[:, j])
                        ),
                        "q25": float(np.quantile(normalized_density[:, j], 0.25)),
                        "q75": float(np.quantile(normalized_density[:, j], 0.75)),
                    }
                )
    return (pd.DataFrame(rows), pd.DataFrame(profiles))


def fixed_budget_selection(predictions: pd.DataFrame) -> pd.DataFrame:
    """Post-inspection, equal-selection-count ranking; not a real-time threshold."""
    rows = []
    for (min_hours, lead, tail), x in predictions.groupby(
        ["min_hours", "lead_h", "tail"], sort=True
    ):
        y = x.event.to_numpy(int)
        p0 = x.p_calendar_total.to_numpy(float)
        p1 = x.p_calendar_total_f82.to_numpy(float)
        p82 = x.p_calendar_N82.to_numpy(float)
        for fraction in (0.05, 0.1, 0.2):
            count = math.ceil(fraction * len(y))
            ix0 = np.argsort(-p0, kind="stable")[:count]
            ix1 = np.argsort(-p1, kind="stable")[:count]
            ix82 = np.argsort(-p82, kind="stable")[:count]
            hits0, hits1, hits82 = (
                int(y[ix0].sum()),
                int(y[ix1].sum()),
                int(y[ix82].sum()),
            )
            record = {
                "min_hours": int(min_hours),
                "lead_h": int(lead),
                "tail": tail,
                "selection_fraction": fraction,
                "n_test": int(len(y)),
                "n_events": int(y.sum()),
                "selected_each_model": count,
                "hits_total": hits0,
                "hits_size": hits1,
                "hits_N82": hits82,
                "additional_true_states": hits1 - hits0,
                "precision_total": hits0 / count,
                "precision_size": hits1 / count,
                "precision_N82": hits82 / count,
                "recall_total": hits0 / y.sum(),
                "recall_size": hits1 / y.sum(),
                "status": "post-inspection retrospective equal-budget ranking",
            }
            if fraction == 0.1:
                weeks = pd.DatetimeIndex(x.block_start).asi8 // 86400000000000 // 7
                unique = np.unique(weeks)
                positions = {week: np.flatnonzero(weeks == week) for week in unique}
                rng = np.random.default_rng(
                    SEED + 100 * lead + 10 * min_hours + len(tail)
                )
                precision_gain = []
                for _ in range(1000):
                    sampled = rng.choice(unique, size=len(unique), replace=True)
                    ix = np.concatenate([positions[week] for week in sampled])
                    take = math.ceil(0.1 * len(ix))
                    selected0 = ix[np.argsort(-p0[ix], kind="stable")[:take]]
                    selected1 = ix[np.argsort(-p1[ix], kind="stable")[:take]]
                    precision_gain.append(
                        (y[selected1].sum() - y[selected0].sum()) / take
                    )
                (
                    record["precision_gain_lo95_7day"],
                    record["precision_gain_hi95_7day"],
                ) = (float(v) for v in np.quantile(precision_gain, [0.025, 0.975]))
            rows.append(record)
    return pd.DataFrame(rows)


def state_year_replay(predictions: pd.DataFrame) -> pd.DataFrame:
    """Describe whether the SGP state gain is confined to one test year."""
    frame = predictions.copy()
    frame["year"] = pd.DatetimeIndex(frame.block_start).year
    rows = []
    for (minimum, lead, tail, year), x in frame.groupby(
        ["min_hours", "lead_h", "tail", "year"]
    ):
        y = x.event.to_numpy(int)
        p0 = x.p_calendar_total.to_numpy(float)
        p1 = x.p_calendar_total_f82.to_numpy(float)
        rows.append(
            {
                "min_hours": int(minimum),
                "lead_h": int(lead),
                "tail": tail,
                "test_year": int(year),
                "n_test": int(len(y)),
                "events": int(y.sum()),
                "brier_total": float(np.mean((y - p0) ** 2)),
                "brier_size": float(np.mean((y - p1) ** 2)),
                "brier_gain": float(np.mean((y - p0) ** 2 - (y - p1) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    OUT.mkdir(parents=True, exist_ok=True, mode=448)
    seasonal, source_files = seasonal_replay()
    d = audited.read_site("SGP", leads=(1, 3, 6))
    blocks, scores, gains, predictions, extras = six_hour_state(d)
    physical, profiles = physical_contrast(d)
    equal_budget = fixed_budget_selection(predictions)
    years = state_year_replay(predictions)
    seasonal.to_csv(OUT / "site_season_lead_scores.csv", index=False)
    blocks.to_csv(OUT / "sgp_six_hour_target_blocks.csv", index=False)
    scores.to_csv(OUT / "sgp_six_hour_state_scores.csv", index=False)
    gains.to_csv(OUT / "sgp_six_hour_state_gains.csv", index=False)
    predictions.to_csv(
        OUT / "sgp_six_hour_state_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    extras["monthly_threshold_support"].to_csv(
        OUT / "sgp_fit_month_thresholds.csv", index=False
    )
    extras["seasonal_state_scores"].to_csv(
        OUT / "sgp_state_season_scores.csv", index=False
    )
    physical.to_csv(OUT / "sgp_size_stratified_physical_contrast.csv", index=False)
    profiles.to_csv(OUT / "sgp_size_stratified_spectra.csv", index=False)
    equal_budget.to_csv(OUT / "sgp_equal_budget_state_selection.csv", index=False)
    years.to_csv(OUT / "sgp_state_year_scores.csv", index=False)
    manifest = extras["manifest"] | {
        "analysis_code_file": file_info(PROTOCOL),
        "hourly_prediction_files": source_files,
        "seasonal_replay_leads": list(LEADS),
        "six_hour_state_leads": list(BLOCK_LEADS),
        "season_support_rule": ">=100 test hours and >=30 test dates",
        "physical_group_definition": "fit-season middle 40% N then fit f82 lower/upper quartiles",
        "physical_group_scope": "observational association, not fixed-N intervention",
        "equal_budget_status": "added after seeing unequal test alarm counts; not a prespecified threshold",
        "N82_model_status": "post-inspection physical-count comparator on the same test blocks",
        "seven_day_bootstrap": "fixed predictions, 1000/2000 draws; no model refit",
        "outputs": sorted((p.name for p in OUT.glob("*.csv*"))),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "season cells",
        len(seasonal),
        "SGP six-hour rows",
        len(scores),
        "physical groups",
        len(physical),
        flush=True,
    )


if __name__ == "__main__":
    main()
