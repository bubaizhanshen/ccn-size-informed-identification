"""Time-separated CCN information on the audited, instrument-screened archive.

The primary eligibility for each lead is a valid current PNSD and a measured
future CCN. No concurrent CCN or future PNSD is required for that comparison.
All leads share the frozen calendar test boundary; a common-origin replay and
an alternative chronological split test sensitivity to eligibility and split.
"""

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
import argparse

import json
import numpy as np
import pandas as pd
from analysis.preprocessing import complete_cohort as builder
from analysis.diagnostics import size_fraction as previous

OUT = ROOT / "results/temporal_information"
GRAPH = ROOT / "results/mapped"
OLD = ROOT / "results/complete_cohort"
LEADS = (1, 3, 6)
SEED = 270927


def read_site(site: str, leads: tuple[int, ...] = LEADS) -> dict:
    if (
        3 not in leads
        or not leads
        or any((h <= 0 for h in leads))
        or (len(set(leads)) != len(leads))
    ):
        raise ValueError(
            "leads must be distinct positive hours and include the frozen 3 h audit"
        )
    manifest = json.loads((OLD / f"{site}_cohort_manifest.json").read_text())
    audit = json.loads((GRAPH / f"{site}_mapped24_audit.json").read_text())
    for key in (
        "preprocessing_gate_passed",
        "conservation_gate_passed",
        "supplied_total_gate_passed",
    ):
        if not audit.get(key):
            raise ValueError(f"{site}: required PNSD gate {key} failed")
    paths = builder.source_smps(site)
    if {p.name: file_info(p) for p in paths} != manifest["source_csv_files"]:
        raise ValueError(f"{site}: input SMPS source changed")
    if (
        file_info(GRAPH / f"{site}_mapped24_audit.json")
        != manifest["mapped_audit_file"]
    ):
        raise ValueError(f"{site}: mapped PNSD audit changed")
    frames, timestamps = ([], [])
    for path in paths:
        raw = pd.read_csv(path, low_memory=False)
        ts = pd.to_datetime(raw.Start_date, errors="coerce", utc=True)
        timestamps.append(ts)
        frame = builder.ccn.harmonized.pnsd_band_values(raw)
        for col in ("P_SMPS", "T_SMPS", "RH_SMPS"):
            frame[col] = pd.to_numeric(raw[col], errors="coerce")
        frame["timestamp"] = ts
        frame["pnsd_base_ok"] = builder.qc_ok(
            pd.to_numeric(raw.Qc_CPC_SMPS, errors="coerce")
        )
        frame["native_spectrum_complete"] = (
            frame[builder.ccn.SPECTRUM].notna().all(axis=1)
        )
        frames.append(frame)
    smps = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    raw_index = pd.Series(
        np.arange(sum((len(t) for t in timestamps))),
        index=pd.DatetimeIndex(pd.concat(timestamps, ignore_index=True)),
    )
    raw_index = raw_index[~raw_index.index.duplicated(keep="last")]
    with np.load(GRAPH / f"{site}_mapped24.npz") as mapped:
        all_bins = mapped["pnsd_bin_number_concentration"].astype(float)
        valid = mapped["valid_hour"].astype(bool)
        diameter = mapped["diameter_midpoint_nm"].astype(float)
    if len(all_bins) != len(raw_index.index) + raw_index.index.duplicated().sum():
        if len(all_bins) != sum((len(t) for t in timestamps)):
            raise ValueError(f"{site}: mapped spectrum row count changed")
    valid &= (
        np.isfinite(all_bins).all(axis=1)
        & (all_bins >= 0).all(axis=1)
        & (all_bins.sum(axis=1) > 0)
    )
    ts = pd.DatetimeIndex(smps.timestamp)
    row = raw_index.reindex(ts).fillna(-1).to_numpy(int)
    if (row < 0).any():
        raise ValueError(f"{site}: missing current spectrum mapping")
    bins = all_bins[row]
    current_ok = (
        valid[row]
        & smps.pnsd_base_ok.to_numpy(bool)
        & smps.native_spectrum_complete.to_numpy(bool)
    )
    ratio = bins[current_ok].sum(axis=1) / pd.to_numeric(
        smps.loc[current_ok, "n_15_300"], errors="coerce"
    ).to_numpy(float)
    q05, q50, q95 = np.nanquantile(ratio, [0.05, 0.5, 0.95])
    if not (0.95 <= q05 <= 1.05 and 0.97 <= q50 <= 1.03 and (0.95 <= q95 <= 1.05)):
        raise ValueError(f"{site}: mapped total/frozen total mismatch")
    total = bins.sum(axis=1)
    n82, f82 = previous.fraction_above(bins[current_ok], diameter, 82.0)
    count82 = np.full(len(smps), np.nan)
    fraction82 = np.full(len(smps), np.nan)
    count82[current_ok], fraction82[current_ok] = (n82, f82)
    hour = ts.hour + ts.minute / 60
    doy = ts.dayofyear
    smps["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    smps["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    smps["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    smps["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    aux = np.column_stack(
        (
            builder.ccn.make_features(smps, builder.ccn.BASE).to_numpy(float),
            np.log1p(total),
        )
    )
    ccn = builder.load_ccn_instrument_only()[site].set_index("timestamp")
    if ccn.index.has_duplicates:
        raise ValueError(f"{site}: duplicate CCN hourly timestamps")
    y, eligible = ({}, {})
    for lead in (0,) + leads:
        values = ccn.reindex(ts + pd.Timedelta(hours=lead))
        measured = pd.to_numeric(values.N_CCN_mean_STP_B, errors="coerce").to_numpy(
            float
        )
        good = (
            np.isfinite(measured)
            & (measured >= 0)
            & values.ccn_instrument_ok.astype("boolean").fillna(False).to_numpy(bool)
        )
        y[lead] = np.log1p(np.clip(measured, 0, None))
        eligible[lead] = current_ok & good
    cutoff = pd.Timestamp(manifest["parent_first_test"])
    future_bins3 = all_bins[
        raw_index.reindex(ts + pd.Timedelta(hours=3))
        .fillna(-1)
        .clip(lower=0)
        .to_numpy(int)
    ]
    future_row3 = raw_index.reindex(ts + pd.Timedelta(hours=3)).fillna(-1).to_numpy(int)
    future_ok3 = future_row3 >= 0
    future_ok3[future_ok3] &= valid[future_row3[future_ok3]]
    complete3 = eligible[3] & eligible[0] & future_ok3
    fit3 = complete3 & np.asarray(ts + pd.Timedelta(hours=3) < cutoff)
    test3 = complete3 & np.asarray(ts >= cutoff)
    with np.load(OLD / f"{site}_cohort.npz") as frozen:
        if int(fit3.sum()) != int(frozen["fit"].sum()) or int(test3.sum()) != int(
            frozen["test"].sum()
        ):
            raise ValueError(f"{site}: complete +3 h cohort not reproduced")
        if not np.array_equal(
            ts[test3].strftime("%Y-%m-%dT%H:%M:%SZ"),
            frozen["timestamp"][frozen["test"].astype(bool)],
        ):
            raise ValueError(f"{site}: complete +3 h test timestamps changed")
    return dict(
        site=site,
        ts=ts,
        bins=bins,
        diameter=diameter,
        current_ok=current_ok,
        total=total,
        n82=count82,
        f82=fraction82,
        aux=aux,
        y=y,
        eligible=eligible,
        cut=cutoff,
        complete3=complete3,
        future_bins3=future_bins3,
        mapped_bins=all_bins,
        mapped_valid=valid,
        raw_index=raw_index,
        source_files=manifest["source_csv_files"],
        mapped_audit_file=manifest["mapped_audit_file"],
        conservation=[float(q05), float(q50), float(q95)],
    )


def fit_predict(
    x: np.ndarray, y: np.ndarray, fit: np.ndarray, test: np.ndarray
) -> np.ndarray:
    keep = np.isfinite(x[fit]).any(axis=0)
    model = previous.regressor().fit(x[fit][:, keep], y[fit])
    return model.predict(x[test][:, keep])


def block_interval(
    y: np.ndarray,
    base: np.ndarray,
    added: np.ndarray,
    dates: np.ndarray,
    days: int,
    seed: int,
    draws: int = 1000,
) -> tuple[float, float]:
    """Nonoverlapping calendar-block bootstrap; predictions remain fitted/fixed."""
    day = pd.to_datetime(dates).astype("int64") // 86400000000000
    block = (day - day.min()) // days
    _, ix = np.unique(block, return_inverse=True)
    nb = int(ix.max() + 1)
    n = np.bincount(ix)
    sy = np.bincount(ix, weights=y)
    sy2 = np.bincount(ix, weights=y * y)
    e0 = np.bincount(ix, weights=(y - base) ** 2)
    e1 = np.bincount(ix, weights=(y - added) ** 2)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, nb, size=(draws, nb))
    count = n[draw].sum(axis=1)
    sst = sy2[draw].sum(axis=1) - sy[draw].sum(axis=1) ** 2 / count
    gain = (e0[draw].sum(axis=1) - e1[draw].sum(axis=1)) / sst
    gain = gain[np.isfinite(gain)]
    return tuple(np.quantile(gain, [0.025, 0.975]))


def site_analysis(
    d: dict, out: Path
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    site, ts, cut = (d["site"], d["ts"], d["cut"])
    xsets = {
        "total": d["aux"],
        "total_plus_f82": np.column_stack((d["aux"], d["f82"])),
        "N82": np.column_stack((d["aux"][:, :-1], np.log1p(d["n82"]))),
    }
    scores, gains, seasonal, alt_splits, physical = ([], [], [], [], [])
    predictions = {}
    common = np.logical_and.reduce([d["eligible"][h] for h in LEADS])
    future_counts = {}
    physical_common = common.copy()
    for lead in LEADS:
        row = (
            d["raw_index"]
            .reindex(ts + pd.Timedelta(hours=lead))
            .fillna(-1)
            .to_numpy(int)
        )
        future_ok = row >= 0
        future_ok[future_ok] &= d["mapped_valid"][row[future_ok]]
        physical_common &= future_ok
        count = np.full(len(ts), np.nan)
        if future_ok.any():
            count[future_ok], _ = previous.fraction_above(
                d["mapped_bins"][row[future_ok]], d["diameter"], 82.0
            )
        future_counts[lead] = count
    for lead in LEADS:
        y = d["y"][lead]
        fit = d["eligible"][lead] & np.asarray(ts + pd.Timedelta(hours=lead) < cut)
        test = d["eligible"][lead] & np.asarray(ts >= cut)
        fit_common = common & np.asarray(ts + pd.Timedelta(hours=max(LEADS)) < cut)
        test_common = common & np.asarray(ts >= cut)
        for cohort, train, evaluate in (
            ("eligible", fit, test),
            ("common_origin", fit_common, test_common),
        ):
            if train.sum() < 100 or evaluate.sum() < 100:
                raise ValueError(f"{site}: too few {cohort} hours at {lead} h")
            preds = {
                name: fit_predict(x, y, train, evaluate) for name, x in xsets.items()
            }
            target = y[evaluate]
            for name, pred in preds.items():
                scores.append(
                    dict(
                        site=site,
                        lead_h=lead,
                        cohort=cohort,
                        model=name,
                        n_fit=int(train.sum()),
                        n_test=int(evaluate.sum()),
                        n_test_dates=int(len(np.unique(ts[evaluate].date))),
                        r2_log=previous.r2(target, pred),
                        mae_log=float(np.mean(np.abs(target - pred))),
                    )
                )
            dates = np.asarray(ts[evaluate].date, dtype=str)
            base, shape = (preds["total"], preds["total_plus_f82"])
            gain, lo, hi = previous.paired_delta(
                target, base, shape, dates, SEED + lead
            )
            row = dict(
                site=site,
                lead_h=lead,
                cohort=cohort,
                n_fit=int(train.sum()),
                n_test=int(evaluate.sum()),
                delta_r2=gain,
                lo95_day=lo,
                hi95_day=hi,
            )
            for days in (3, 7):
                row[f"lo95_{days}day"], row[f"hi95_{days}day"] = block_interval(
                    target, base, shape, dates, days, SEED + 100 * days + lead
                )
            gains.append(row)
            if cohort == "eligible":
                predictions[lead] = pd.DataFrame(
                    dict(
                        timestamp=ts[evaluate].strftime("%Y-%m-%dT%H:%M:%SZ"),
                        date=dates,
                        month=ts[evaluate].month,
                        target_log=target,
                        current_ccn_log=d["y"][0][evaluate],
                        current_ccn_available=d["eligible"][0][evaluate],
                        **{f"pred_{name}": p for name, p in preds.items()},
                    )
                )
                if site == "SGP":
                    seasons = np.array(["DJF", "MAM", "JJA", "SON"])[
                        ts[evaluate].month % 12 // 3
                    ]
                    for season in ("DJF", "MAM", "JJA", "SON"):
                        m = seasons == season
                        if m.sum() < 100:
                            continue
                        seasonal.append(
                            dict(
                                site=site,
                                lead_h=lead,
                                season=season,
                                n_test=int(m.sum()),
                                n_test_dates=int(len(np.unique(dates[m]))),
                                r2_total=previous.r2(target[m], base[m]),
                                r2_size=previous.r2(target[m], shape[m]),
                                delta_r2=previous.r2(target[m], shape[m])
                                - previous.r2(target[m], base[m]),
                            )
                        )
        physical_test = physical_common & np.asarray(ts >= cut)
        now = np.log1p(d["n82"][physical_test])
        later_count = np.log1p(future_counts[lead][physical_test])
        later_ccn = y[physical_test]
        physical.append(
            dict(
                site=site,
                lead_h=lead,
                cohort="common_physical_origins",
                n_test=int(physical_test.sum()),
                r2_N82_persistence_identity=previous.r2(later_count, now),
                r2_future_CCN_from_future_N82_identity=previous.r2(
                    later_ccn, later_count
                ),
                r2_future_CCN_from_current_N82_identity=previous.r2(later_ccn, now),
            )
        )
    lead = 3
    mask = d["eligible"][3] & d["eligible"][0]
    fit = mask & np.asarray(ts + pd.Timedelta(hours=3) < cut)
    test = mask & np.asarray(ts >= cut)
    if fit.sum() < 100 or test.sum() < 100:
        raise ValueError(f"{site}: too few current-CCN control rows")
    x0 = np.column_stack((d["aux"], d["y"][0]))
    x1 = np.column_stack((d["aux"], d["y"][0], d["f82"]))
    p0, p1 = (fit_predict(x, d["y"][3], fit, test) for x in (x0, x1))
    target = d["y"][3][test]
    dates = np.asarray(ts[test].date, dtype=str)
    delta, lo, hi = previous.paired_delta(target, p0, p1, dates, SEED + 500)
    control = dict(
        site=site,
        cohort="current_PNSD_current_CCN_future_CCN",
        n_fit=int(fit.sum()),
        n_test=int(test.sum()),
        r2_current_ccn=previous.r2(target, p0),
        r2_plus_f82=previous.r2(target, p1),
        delta_r2=delta,
        lo95_day=lo,
        hi95_day=hi,
    )
    origin_dates = np.unique(ts[d["current_ok"]].date)
    alt_cut = pd.Timestamp(str(origin_dates[int(0.7 * len(origin_dates))]), tz="UTC")
    fit = d["eligible"][3] & np.asarray(ts + pd.Timedelta(hours=3) < alt_cut)
    test = d["eligible"][3] & np.asarray(ts >= alt_cut)
    if fit.sum() < 100 or test.sum() < 100:
        raise ValueError(f"{site}: alternative split too small")
    p0, p1 = (
        fit_predict(x, d["y"][3], fit, test)
        for x in (xsets["total"], xsets["total_plus_f82"])
    )
    target = d["y"][3][test]
    dates = np.asarray(ts[test].date, dtype=str)
    delta, lo, hi = previous.paired_delta(target, p0, p1, dates, SEED + 700)
    alt_splits.append(
        dict(
            site=site,
            split="70pct_valid_origin_dates",
            test_boundary=str(alt_cut),
            n_fit=int(fit.sum()),
            n_test=int(test.sum()),
            r2_total=previous.r2(target, p0),
            r2_size=previous.r2(target, p1),
            delta_r2=delta,
            lo95_day=lo,
            hi95_day=hi,
        )
    )
    for lead, frame in predictions.items():
        frame.to_csv(out / f"{site}_{lead}h_predictions.csv.gz", index=False)
    manifest = dict(
        site=site,
        status="passed",
        leads_h=LEADS,
        source_files=d["source_files"],
        mapped_audit_file=d["mapped_audit_file"],
        mapped_total_over_frozen_q05_median_q95=d["conservation"],
        frozen_test_boundary=str(cut),
        complete_3h_hours=int(d["complete3"].sum()),
        alternative_boundary=str(alt_cut),
        interpretation="held-out predictive information, not a causal or universal aerosol lifetime",
    )
    (out / f"{site}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return (scores, gains, seasonal, alt_splits, physical, control)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=previous.SITES)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True, mode=448)
    sites = (args.site,) if args.site else previous.SITES
    for site in sites:
        d = read_site(site)
        scores, gains, seasonal, alt_splits, physical, control = site_analysis(d, OUT)
        for name, data in (
            ("scores", scores),
            ("gains", gains),
            ("seasonal", seasonal),
            ("alternative_split", alt_splits),
            ("physical", physical),
            ("current_ccn_control", [control]),
        ):
            pd.DataFrame(data).to_csv(OUT / f"{site}_{name}.csv", index=False)
        print(
            site,
            "done",
            "minimal +3 h test",
            next(
                (
                    r["n_test"]
                    for r in gains
                    if r["lead_h"] == 3 and r["cohort"] == "eligible"
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
