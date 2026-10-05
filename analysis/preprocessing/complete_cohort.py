"""Rebuild the nine-site CCN/PNSD cohort without the CCN--N80 closure flag.

The existing size-graph cohort applied a closure flag whose definition uses
the very CCN--N80 relationship tested here. This cohort keeps instrument QC
and valid spectra/targets, but never filters on that relationship.
"""

from __future__ import annotations

from analysis.common.files import file_info
import argparse

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.common import ccn_archive as ccn
from analysis.common.runtime import ROOT

GRAPH = ROOT / "results/mapped"
OUT = ROOT / "results/complete_cohort"
SOURCE = ROOT / "inputs/figshare_27913806/selected"
SITES = ("ANX", "COR", "ENA", "GUC", "MAO", "MOS", "SBS_CP", "SBS_SPL", "SGP")


def qc_ok(q: pd.Series) -> pd.Series:
    return q.isna() | (q == 0)


def load_ccn_instrument_only() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for path in sorted((SOURCE / "ccn").glob("*_ccn_colb_hour*.csv")):
        match = re.match("([A-Za-z]+)_ccn_colb_hour", path.name)
        if not match:
            continue
        site = match.group(1).upper()
        x = pd.read_csv(
            path,
            usecols=[
                "Start_date",
                "SS_setpoint_B",
                "N_CCN_mean_STP_B",
                "Qc_CPC_SMPS",
                "Qc_CCN04_N80_B",
            ],
            low_memory=False,
        )
        x["timestamp"] = pd.to_datetime(x["Start_date"], errors="coerce", utc=True)
        for col in [
            "SS_setpoint_B",
            "N_CCN_mean_STP_B",
            "Qc_CPC_SMPS",
            "Qc_CCN04_N80_B",
        ]:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        x = x[np.isclose(x.SS_setpoint_B, 0.4, atol=1e-06)].copy()
        x["ccn_instrument_ok"] = qc_ok(x["Qc_CPC_SMPS"])
        x["ccn_closure_ok"] = qc_ok(x["Qc_CCN04_N80_B"])
        x = x[["timestamp", "N_CCN_mean_STP_B", "ccn_instrument_ok", "ccn_closure_ok"]]
        result[site] = pd.concat([result.get(site, x.iloc[:0]), x], ignore_index=True)
    result = {
        site: frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        for site, frame in result.items()
    }
    for path in sorted((SOURCE / "ccn_cola").glob("*_ccn_cola.csv")):
        match = re.match("([A-Za-z_]+)_ccn_cola", path.name)
        if not match:
            continue
        site = match.group(1).upper()
        if site in result:
            continue
        x = pd.read_csv(
            path,
            usecols=[
                "Start_date",
                "SS_harmonized_A",
                "N_CCN_mean_STP_A",
                "Qc_CCN04_N80_A",
            ],
            low_memory=False,
        )
        x["timestamp"] = pd.to_datetime(
            x["Start_date"], errors="coerce", utc=True
        ).dt.floor("h")
        for col in ["SS_harmonized_A", "N_CCN_mean_STP_A", "Qc_CCN04_N80_A"]:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        x = x[np.isclose(x.SS_harmonized_A, 0.4, atol=1e-06)].copy()
        x["ccn_closure_ok"] = qc_ok(x["Qc_CCN04_N80_A"])
        grouped = x.groupby("timestamp", sort=True)
        frame = grouped["N_CCN_mean_STP_A"].mean().rename("N_CCN_mean_STP_B").to_frame()
        frame["ccn_closure_ok"] = grouped["ccn_closure_ok"].all()
        frame["ccn_instrument_ok"] = True
        result[site] = frame.reset_index().drop_duplicates("timestamp", keep="last")
    return result


def source_smps(site: str) -> list[Path]:
    token = {"SBS_CP": "SBS_cp", "SBS_SPL": "SBS_spl"}.get(site, site)
    return sorted((SOURCE / "smps").glob(f"{token}_smps_hour*.csv"))


def prepare(
    site: str, smps: pd.DataFrame, ccn_data: pd.DataFrame, output: Path
) -> dict:
    prior = json.loads(
        (GRAPH / f"{site}_cohort_manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (GRAPH / f"{site}_mapped24_audit.json").read_text(encoding="utf-8")
    )
    if not audit.get("preprocessing_gate_passed"):
        raise ValueError(f"{site}: mapped 24-bin audit failed")
    paths = source_smps(site)
    if {path.name: file_info(path) for path in paths} != prior["source_csv_files"]:
        raise ValueError(f"{site}: source SMPS file changed")
    with np.load(GRAPH / f"{site}_mapped24.npz") as mapped:
        bins = mapped["pnsd_bin_number_concentration"].astype(np.float32)
        valid = mapped["valid_hour"].astype(bool)
        grid = mapped["diameter_midpoint_nm"].astype(np.float32)
    raw_times = pd.to_datetime(
        pd.concat(
            [pd.read_csv(p, usecols=["Start_date"]) for p in paths], ignore_index=True
        )["Start_date"],
        errors="coerce",
        utc=True,
    )
    if len(raw_times) != len(bins):
        raise ValueError(f"{site}: mapped rows and raw SMPS timestamps differ")
    mapped_index = pd.Series(np.arange(len(bins)), index=pd.DatetimeIndex(raw_times))
    mapped_index = mapped_index[~mapped_index.index.duplicated(keep="last")]
    x = smps.merge(ccn_data, on="timestamp", how="inner", validate="one_to_one")
    x = x.sort_values("timestamp").reset_index(drop=True)
    hour = x.timestamp.dt.hour + x.timestamp.dt.minute / 60
    doy = x.timestamp.dt.dayofyear
    x["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    x["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    x["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    x["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    x["date"] = x.timestamp.dt.strftime("%Y-%m-%d")
    x["future_timestamp"] = x.timestamp + pd.Timedelta(hours=3)
    future = x[
        [
            "timestamp",
            "N_CCN_mean_STP_B",
            "ccn_instrument_ok",
            "ccn_closure_ok",
            "pnsd_base_ok",
        ]
    ].rename(
        columns={
            "timestamp": "future_timestamp",
            "N_CCN_mean_STP_B": "future_ccn",
            "ccn_instrument_ok": "future_ccn_instrument_ok",
            "ccn_closure_ok": "future_ccn_closure_ok",
            "pnsd_base_ok": "future_pnsd_base_ok",
        }
    )
    x = x.merge(future, on="future_timestamp", how="left", validate="many_to_one")
    i_now = mapped_index.reindex(pd.DatetimeIndex(x.timestamp)).fillna(-1).to_numpy(int)
    i_future = (
        mapped_index.reindex(pd.DatetimeIndex(x.future_timestamp))
        .fillna(-1)
        .to_numpy(int)
    )
    now = np.full((len(x), 24), np.nan, dtype=np.float32)
    after = np.full_like(now, np.nan)
    have_now, have_future = (i_now >= 0, i_future >= 0)
    now[have_now], after[have_future] = (
        bins[i_now[have_now]],
        bins[i_future[have_future]],
    )
    okay = have_now & have_future
    okay[have_now] &= valid[i_now[have_now]]
    okay[have_future] &= valid[i_future[have_future]]
    okay &= np.isfinite(now).all(axis=1) & np.isfinite(after).all(axis=1)
    okay &= (now.sum(axis=1) > 0) & (after.sum(axis=1) > 0)
    for key in ("pnsd_base_ok", "ccn_instrument_ok", "future_ccn_instrument_ok"):
        okay &= x[key].astype("boolean").fillna(False).to_numpy(dtype=bool)
    current_ccn = pd.to_numeric(x.N_CCN_mean_STP_B, errors="coerce").to_numpy(float)
    future_ccn = pd.to_numeric(x.future_ccn, errors="coerce").to_numpy(float)
    okay &= np.isfinite(current_ccn) & (current_ccn >= 0)
    okay &= np.isfinite(future_ccn) & (future_ccn >= 0)
    okay &= x[ccn.SPECTRUM].notna().all(axis=1).to_numpy()
    cut = pd.Timestamp(prior["first_test"])
    fit = okay & (x.future_timestamp.to_numpy() < cut)
    test = okay & (x.timestamp.to_numpy() >= cut)
    if fit.sum() < 100 or test.sum() < 100:
        raise ValueError(f"{site}: insufficient instrument-only fit/test rows")
    chosen = fit | test
    x = x.loc[chosen].reset_index(drop=True)
    fit, test = (fit[chosen], test[chosen])
    now, after = (now[chosen], after[chosen])
    current_ccn, future_ccn = (current_ccn[chosen], future_ccn[chosen])
    ratio = now.sum(axis=1) / pd.to_numeric(x["n_15_300"], errors="coerce").to_numpy(
        float
    )
    q05, q50, q95 = np.nanquantile(ratio, [0.05, 0.5, 0.95])
    if not (0.95 <= q05 <= 1.05 and 0.97 <= q50 <= 1.03 and (0.95 <= q95 <= 1.05)):
        raise ValueError(f"{site}: mapped total fails frozen 15–300 nm agreement")
    closure_now = x.ccn_closure_ok.astype("boolean").fillna(False).to_numpy(dtype=bool)
    closure_future = (
        x.future_ccn_closure_ok.astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    aux = np.column_stack(
        (ccn.make_features(x, ccn.BASE).to_numpy(np.float32), np.log1p(now.sum(axis=1)))
    ).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    cohort_file = output / f"{site}_cohort.npz"
    np.savez_compressed(
        cohort_file,
        bins_now=now,
        bins_future=after,
        current_ccn=np.log1p(current_ccn).astype(np.float32),
        y_ccn=np.log1p(future_ccn).astype(np.float32),
        aux=aux,
        diameter_midpoint_nm=grid,
        fit=fit,
        val=np.zeros(len(fit), dtype=bool),
        test=test,
        date=x.date.to_numpy(dtype=str),
        timestamp=x.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(dtype=str),
        current_closure_ok=closure_now,
        future_closure_ok=closure_future,
    )
    info = {
        "site": site,
        "cohort": "instrument-only; no CCN--N80 closure filter",
        "qc": "current/future CCN CPC instrument flag and current SMPS CPC flag missing or zero; finite nonnegative CCN; complete mapped current/future PNSD; same future SMPS flag policy as graph cohort",
        "new_preflight_passed": True,
        "source_csv_files": prior["source_csv_files"],
        "mapped_audit_file": file_info(GRAPH / f"{site}_mapped24_audit.json"),
        "mapped_total_over_frozen_total_q05_median_q95": [
            float(q05),
            float(q50),
            float(q95),
        ],
        "parent_first_test": prior["first_test"],
        "split": "same first test timestamp as closure-selected graph cohort; fit labels end before test timestamp",
        "n_fit": int(fit.sum()),
        "n_val": 0,
        "n_test": int(test.sum()),
        "n_test_dates": int(len(np.unique(x.date.to_numpy()[test]))),
        "test_closure_pass_both": int((closure_now[test] & closure_future[test]).sum()),
        "test_closure_fail_either": int(
            (~(closure_now[test] & closure_future[test])).sum()
        ),
    }
    (output / f"{site}_cohort_manifest.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    smps = ccn.harmonized.load_smps()
    ccn_data = load_ccn_instrument_only()
    for site in SITES:
        if site not in smps or site not in ccn_data:
            raise ValueError(f"{site}: missing SMPS or CCN records")
        info = prepare(site, smps[site], ccn_data[site], args.output)
        print(
            site,
            info["n_fit"],
            info["n_test"],
            info["test_closure_fail_either"],
            flush=True,
        )


if __name__ == "__main__":
    main()
