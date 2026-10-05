"""Frozen SGP meteorological controls; original data, matched samples, no future met."""

from analysis.common.files import file_info

import os
from pathlib import Path
from analysis.common.runtime import ROOT
from analysis.common.runtime import WORK

WORK.mkdir(parents=True, exist_ok=True)
for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr
import sklearn
from analysis.primary import cross_site as ref

RAW = ROOT / "inputs/arm_sgpmetE13_b1"
OLD = ROOT / "results/cross_site/SGP"
OUT = ROOT / "results/meteorology"
OUT.mkdir(parents=True, exist_ok=True)
VARS = [
    "temp_mean",
    "rh_mean",
    "atmos_pressure",
    "wspd_arith_mean",
    "wspd_vec_mean",
    "wdir_vec_mean",
    "tbrg_precip_total",
]
UNITS = ["degC", "%", "kPa", "m/s", "m/s", "degree", "mm"]
CORE = ["temperature", "rh", "pressure", "windspeed", "u", "v"]


def hourly():
    receipt = {
        "complete": True,
        "files": [
            {"file": p.name, "file_info": file_info(p)}
            for p in sorted(RAW.glob("sgpmetE13.b1.*.nc"))
        ],
    }
    if not receipt["files"]:
        raise FileNotFoundError(
            "No ARM sgpmetE13.b1 daily NetCDF files in input directory"
        )
    expected = set(
        pd.date_range("2017-04-01", "2023-10-16", freq="D").strftime("%Y%m%d")
    )
    present = {item["file"].split(".")[2] for item in receipt["files"]}
    missing = expected - present
    if missing:
        raise ValueError(
            f"ARM daily-file coverage incomplete: {len(missing)} missing days; obtain the complete stated period"
        )
    receipt["files"] = [
        item for item in receipt["files"] if item["file"].split(".")[2] in expected
    ]
    (OUT / "input_manifest.json").write_text(json.dumps(receipt, indent=2))
    assert receipt[
        "complete"
    ], "Wait for the complete download; do not score partial years"
    parts = []
    metadata = []
    for i, item in enumerate(receipt["files"]):
        p = RAW / item["file"]
        assert file_info(p) == item["file_info"]
        with xr.open_dataset(p, decode_times=False) as ds:
            assert int(float(ds.attrs["averaging_interval"].split()[0])) == 60
            assert "end" in ds.attrs["averaging_interval_comment"].lower()
            t = xr.coding.times.decode_cf_datetime(
                ds.time.values, ds.time.attrs["units"]
            )
            f = pd.DataFrame(index=pd.DatetimeIndex(t).tz_localize("UTC"))
            assert not f.index.has_duplicates
            for v, u in zip(VARS, UNITS):
                assert ds[v].attrs["units"] == u, (p.name, v, ds[v].attrs["units"])
                qa = ds["qc_" + v].attrs
                for bit in (1, 2, 3):
                    assessment = qa.get(
                        f"bit_{bit}_assessment",
                        ds.attrs.get(f"qc_bit_{bit}_assessment"),
                    )
                    assert str(assessment).lower() == "bad", (p.name, v, bit)
                delta = qa.get("bit_4_assessment", ds.attrs.get("qc_bit_4_assessment"))
                assert delta is None or str(delta).lower() == "indeterminate"
                a = ds[v].values.astype(float)
                q = ds["qc_" + v].values.astype(float)
                if delta is None:
                    qi = np.nan_to_num(q, nan=-1).astype(np.int64)
                    q[qi & 8 != 0] = -1
                a[
                    (a < ds[v].attrs.get("valid_min", -np.inf))
                    | (a > ds[v].attrs.get("valid_max", np.inf))
                ] = np.nan
                f[v] = a
                f["qc_" + v] = q
            metadata.append(
                dict(
                    file=p.name,
                    dod_version=ds.attrs.get("dod_version"),
                    qc_schema=(
                        "global"
                        if "qc_bit_1_assessment" in ds.attrs
                        else "per-variable"
                    ),
                    n_minutes=len(f),
                    start=str(f.index.min()),
                    end=str(f.index.max()),
                    latitude=float(ds.lat.values),
                    longitude=float(ds.lon.values),
                    altitude_m=float(ds.alt.values),
                )
            )
            parts.append(f)
        if (i + 1) % 200 == 0:
            print("Read meteorology daily files", i + 1, flush=True)
    minutes = pd.concat(parts).sort_index()
    del parts
    duplicate = minutes.index.duplicated(keep=False)
    if duplicate.any():
        for _, g in minutes.loc[duplicate].groupby(level=0):
            assert np.allclose(
                g.to_numpy(), g.iloc[0].to_numpy()[None, :], equal_nan=True
            ), "Conflicting duplicate meteorology"
    n_duplicate = int(minutes.index.duplicated().sum())
    minutes = minutes[~minutes.index.duplicated()]
    assert (
        minutes.index.asi8 % (60 * 10**9) == 0
    ).all(), "Minute time grid requires explicit review"
    pd.DataFrame(metadata).to_csv(OUT / "raw_file_metadata.csv", index=False)
    qcrows = []
    result = {}
    for strict in (False, True):
        name = "strict" if strict else "primary"
        clean = pd.DataFrame(index=minutes.index)
        for v in VARS:
            q = minutes["qc_" + v].fillna(-1).to_numpy().astype(np.int64)
            ok = q == 0 if strict else (q >= 0) & (q & ~8 == 0)
            clean[v] = minutes[v].where(ok)
            qcrows.append(
                dict(
                    qc=name,
                    variable=v,
                    n_minutes=len(clean),
                    n_valid=int(clean[v].notna().sum()),
                )
            )
        a = pd.DataFrame(index=clean.index)
        for v, c in zip(VARS[:4], CORE[:4]):
            a[c] = clean[v]
        a["u"] = -clean.wspd_vec_mean * np.sin(np.deg2rad(clean.wdir_vec_mean))
        a["v"] = -clean.wspd_vec_mean * np.cos(np.deg2rad(clean.wdir_vec_mean))
        r = a.resample("1h", closed="right", label="right")
        count = r.count()
        h = r.mean().where(count >= 45)
        rain = clean.tbrg_precip_total.resample("1h", closed="right", label="right")
        h["rain_mm"] = rain.sum(min_count=60).where(rain.count() == 60)
        h = h.loc[h.index <= pd.Timestamp("2023-10-16 23:00:00", tz="UTC")]
        h.index.name = "origin"
        h.to_csv(OUT / f"hourly_{name}.csv.gz")
        count.to_csv(OUT / f"minute_coverage_{name}.csv.gz")
        result[name] = h
    pd.DataFrame(qcrows).to_csv(OUT / "minute_qc_coverage.csv", index=False)
    return (
        result,
        dict(
            n_daily_files=len(receipt["files"]),
            duplicate_minutes_removed=n_duplicate,
            n_unique_minutes=len(minutes),
            download_manifest_file=file_info(OUT / "input_manifest.json"),
        ),
    )


def main():
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Completed analysis already exists")
    hs, manifest = hourly()
    b = pd.read_csv(OLD / "blocks.csv.gz")
    for c in ["block_start", "block_end", "origin_1h", "origin_3h", "origin_6h"]:
        b[c] = pd.to_datetime(b[c], utc=True)
    thresholds = pd.read_csv(OLD / "monthly_threshold_support.csv").set_index("month")
    th = b.month.map(thresholds.threshold_cm3.where(thresholds.supported)).to_numpy()
    y = (b.mean_ccn_cm3.to_numpy() >= th).astype(int)
    oldmanifest = json.loads((OLD / "manifest.json").read_text())
    assert file_info(OLD / "blocks.csv.gz") == oldmanifest["outputs"]["blocks.csv.gz"]
    assert (
        file_info(OLD / "monthly_threshold_support.csv")
        == oldmanifest["outputs"]["monthly_threshold_support.csv"]
    )
    audited = ref.audited.read_site("SGP")
    assert audited["source_files"] == oldmanifest["source_files"]
    assert audited["mapped_audit_file"] == oldmanifest["mapped_audit_file"]
    for lead in (1, 3, 6):
        good = b[f"origin_valid_{lead}h"].to_numpy(bool)
        ix = audited["ts"].get_indexer(pd.DatetimeIndex(b.loc[good, f"origin_{lead}h"]))
        assert (ix >= 0).all() and audited["current_ok"][ix].all()
        for col, key in [("N15_300", "total"), ("f82", "f82")]:
            assert np.allclose(
                b.loc[good, f"{col}_{lead}h"], audited[key][ix], rtol=1e-12, atol=1e-12
            )
    cut = pd.Timestamp(oldmanifest["calibration_boundary"])
    outer = pd.Timestamp(oldmanifest["outer_boundary"])
    rows = []
    gains = []
    support = []
    seasonal = []
    selections = []
    coverage = []
    season_gains = []
    for lead in (1, 3, 6):
        xs = ref.orig.block_features(b, lead)
        eligible = (
            (b.n_valid_ccn_hours >= 4) & b[f"origin_valid_{lead}h"] & np.isfinite(th)
        )
        fit0 = eligible & (b.block_end <= cut)
        test0 = eligible & (b[f"origin_{lead}h"] >= outer)
        old = pd.read_csv(OLD / f"monthly_lead{lead}_predictions.csv.gz")
        assert np.array_equal(
            pd.to_datetime(old.block_start, utc=True), b.loc[test0, "block_start"]
        )
        for name in ("calendar_total", "calendar_total_f82"):
            p = (
                ref.orig.classifier()
                .fit(xs[name][fit0], y[fit0])
                .predict_proba(xs[name][test0])[:, 1]
            )
            assert np.allclose(
                p, old["p_" + name], atol=1e-10, rtol=0
            ), "Original regression check failed"
        for qc, h in hs.items():
            aligned = h.reindex(pd.DatetimeIndex(b[f"origin_{lead}h"]))
            for rain in (False, True):
                subset = qc + ("_rain" if rain else "_core")
                met = aligned[CORE].to_numpy()
                if rain:
                    met = np.column_stack([met, np.log1p(aligned.rain_mm.to_numpy())])
                valid = np.isfinite(met).all(axis=1)
                fit = fit0 & valid
                test = test0 & valid
                support.append(
                    dict(
                        subset=subset,
                        lead_h=lead,
                        n_fit_original=int(fit0.sum()),
                        n_fit=int(fit.sum()),
                        n_test_original=int(test0.sum()),
                        n_test=int(test.sum()),
                        high_test=int(y[test].sum()),
                    )
                )
                for period, base, matched in [
                    ("fit", fit0, fit),
                    ("test", test0, test),
                ]:
                    for month in range(1, 13):
                        mm = b.month == month
                        coverage.append(
                            dict(
                                subset=subset,
                                lead_h=lead,
                                period=period,
                                month=month,
                                n_original=int((base & mm).sum()),
                                n_matched=int((matched & mm).sum()),
                            )
                        )
                assert (
                    fit.sum() > 40 and test.sum() > 20 and (len(np.unique(y[fit])) == 2)
                )
                xsets = {
                    "N": xs["calendar_total"],
                    "N_f82": xs["calendar_total_f82"],
                    "N_met": np.column_stack([xs["calendar_total"], met]),
                    "N_met_f82": np.column_stack([xs["calendar_total_f82"], met]),
                }
                pred = b.loc[
                    test,
                    ["block_start", "block_end", "month", "season", "mean_ccn_cm3"],
                ].copy()
                pred["origin"] = b.loc[test, f"origin_{lead}h"]
                pred["event"] = y[test]
                pred["threshold_cm3"] = th[test]
                yt = y[test]
                for name, x in xsets.items():
                    pred["p_" + name] = (
                        ref.orig.classifier()
                        .fit(x[fit], y[fit])
                        .predict_proba(x[test])[:, 1]
                    )
                    rows.append(
                        dict(
                            subset=subset,
                            lead_h=lead,
                            model=name,
                            n_fit=int(fit.sum()),
                            **ref.scores(yt, pred["p_" + name].to_numpy()),
                        )
                    )
                    for season, g in pred.groupby("season"):
                        seasonal.append(
                            dict(
                                subset=subset,
                                lead_h=lead,
                                model=name,
                                season=season,
                                **ref.scores(
                                    g.event.to_numpy(), g["p_" + name].to_numpy()
                                ),
                            )
                        )
                for j, (a, c) in enumerate(
                    [("N", "N_f82"), ("N_met", "N_met_f82"), ("N", "N_met")]
                ):
                    pa = pred["p_" + a].to_numpy()
                    pc = pred["p_" + c].to_numpy()
                    gains.append(
                        dict(
                            subset=subset,
                            lead_h=lead,
                            baseline=a,
                            added=c,
                            brier_gain=float(np.mean((yt - pa) ** 2 - (yt - pc) ** 2)),
                            precision_gain=float(
                                yt[ref.rank(pc)].mean() - yt[ref.rank(pa)].mean()
                            ),
                            **ref.intervals(
                                yt, pa, pc, pred.block_start, 928000 + lead * 10 + j
                            ),
                        )
                    )
                    if lead == 3 and j < 2:
                        for si, (season, g) in enumerate(pred.groupby("season")):
                            z = g.event.to_numpy()
                            p0 = g["p_" + a].to_numpy()
                            p1 = g["p_" + c].to_numpy()
                            season_gains.append(
                                dict(
                                    subset=subset,
                                    lead_h=lead,
                                    season=season,
                                    baseline=a,
                                    added=c,
                                    n_test=len(g),
                                    brier_gain=float(
                                        np.mean((z - p0) ** 2 - (z - p1) ** 2)
                                    ),
                                    **ref.intervals(
                                        z, p0, p1, g.block_start, 928500 + si * 10 + j
                                    ),
                                )
                            )
                    s0 = np.zeros(len(yt), bool)
                    s1 = s0.copy()
                    s0[ref.rank(pa)] = True
                    s1[ref.rank(pc)] = True
                    for label, m in [
                        ("both", s0 & s1),
                        ("baseline_only", s0 & ~s1),
                        ("added_only", s1 & ~s0),
                    ]:
                        selections.append(
                            dict(
                                subset=subset,
                                lead_h=lead,
                                baseline=a,
                                added=c,
                                selection=label,
                                n_blocks=int(m.sum()),
                                high_blocks=int(yt[m].sum()),
                                median_ccn_cm3=(
                                    float(pred.loc[m, "mean_ccn_cm3"].median())
                                    if m.any()
                                    else None
                                ),
                            )
                        )
                pred.to_csv(
                    OUT / f"{subset}_lead{lead}_predictions.csv.gz", index=False
                )
                print(
                    subset,
                    lead,
                    "fit/test",
                    int(fit.sum()),
                    int(test.sum()),
                    flush=True,
                )
    for name, records in [
        ("scores", rows),
        ("paired_gains", gains),
        ("eligibility", support),
        ("seasonal_scores", seasonal),
        ("seasonal_paired_gains", season_gains),
        ("selected_set_concentrations", selections),
        ("monthly_coverage", coverage),
    ]:
        pd.DataFrame(records).to_csv(OUT / (name + ".csv"), index=False)
    manifest.update(
        status="complete",
        completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        software=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            pandas=pd.__version__,
            xarray=xr.__version__,
            sklearn=sklearn.__version__,
        ),
        original_predictions_reproduced=True,
        calibration_boundary=str(cut),
        outer_boundary=str(outer),
        script_file=file_info(Path(__file__)),
        analysis_code_file=file_info(Path(__file__)),
        original_manifest_file=file_info(OLD / "manifest.json"),
        outputs={p.name: file_info(p) for p in OUT.iterdir() if p.is_file()},
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
