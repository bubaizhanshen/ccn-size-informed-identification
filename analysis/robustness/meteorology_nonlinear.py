"""Targeted nonlinear met control and fixed-budget CCN selection checks."""

from analysis.common.files import file_info

import os
from pathlib import Path
from analysis.common.runtime import ROOT
from analysis.common.runtime import WORK

os.umask(63)
WORK.mkdir(parents=True, exist_ok=True, mode=448)
for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from analysis.primary import cross_site as ref

for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
OLD = ROOT / "results/cross_site/SGP"
MET = ROOT / "results/meteorology"
OUT = ROOT / "results/meteorology_nonlinear"
OUT.mkdir(parents=True, exist_ok=True)
Q = (0.05, 0.1, 0.2, 0.3)
PARAM = dict(
    n_estimators=300,
    num_leaves=15,
    learning_rate=0.03,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=20260928,
    n_jobs=1,
    verbosity=-1,
    deterministic=True,
    force_col_wise=True,
)


def rank(p, q):
    return np.argsort(-p, kind="stable")[: math.ceil(q * len(p))]


def evaluate(pred, meta, models, rows, gains, budgets, budget_gains, unique):
    y = pred.event.to_numpy()
    c = pred.mean_ccn_cm3.to_numpy()
    dates = pd.DatetimeIndex(pred.block_start)
    for name in models:
        p = pred["p_" + name].to_numpy()
        rows.append(dict(**meta, model=name, **ref.scores(y, p)))
        for q in Q:
            ix = rank(p, q)
            budgets.append(
                dict(
                    **meta,
                    model=name,
                    fraction=q,
                    n_test=len(y),
                    high_blocks=int(y.sum()),
                    selected=len(ix),
                    hits=int(y[ix].sum()),
                    precision=float(y[ix].mean()),
                    recall=float(y[ix].sum() / y.sum()),
                    median_ccn_cm3=float(np.median(c[ix])),
                )
            )
    pairs = [("N", "N_f82")] + (
        [("N_met", "N_met_f82")] if "N_met" in models else [("N", "N82")]
    )
    week = dates.asi8 // (7 * 86400 * 10**9)
    groups = [np.flatnonzero(week == w) for w in np.unique(week)]
    for j, (a, b) in enumerate(pairs):
        p = pred["p_" + a].to_numpy()
        r = pred["p_" + b].to_numpy()
        gains.append(
            dict(
                **meta,
                baseline=a,
                added=b,
                brier_gain=float(np.mean((y - p) ** 2 - (y - r) ** 2)),
                **ref.intervals(y, p, r, dates, 928700 + meta["lead_h"] * 10 + j),
            )
        )
        rng = np.random.default_rng(928800 + meta["lead_h"] * 10 + j)
        draws = {q: [] for q in Q}
        for _ in range(1000):
            ix = np.concatenate(
                [groups[k] for k in rng.integers(len(groups), size=len(groups))]
            )
            op = np.argsort(-p[ix], kind="stable")
            orr = np.argsort(-r[ix], kind="stable")
            for q in Q:
                n = math.ceil(q * len(ix))
                i0 = ix[op[:n]]
                i1 = ix[orr[:n]]
                draws[q].append(
                    (y[i1].mean() - y[i0].mean(), np.median(c[i1]) - np.median(c[i0]))
                )
        for q in Q:
            i0 = rank(p, q)
            i1 = rank(r, q)
            ci = np.quantile(draws[q], [0.025, 0.975], axis=0)
            budget_gains.append(
                dict(
                    **meta,
                    baseline=a,
                    added=b,
                    fraction=q,
                    precision_gain=float(y[i1].mean() - y[i0].mean()),
                    precision_lo95=ci[0, 0],
                    precision_hi95=ci[1, 0],
                    median_ccn_gain_cm3=float(np.median(c[i1]) - np.median(c[i0])),
                    median_lo95=ci[0, 1],
                    median_hi95=ci[1, 1],
                    n_weeks=len(groups),
                    draws=1000,
                )
            )
            s0 = np.zeros(len(y), bool)
            s1 = s0.copy()
            s0[i0] = True
            s1[i1] = True
            for label, m in [
                ("both", s0 & s1),
                ("baseline_only", s0 & ~s1),
                ("added_only", s1 & ~s0),
            ]:
                unique.append(
                    dict(
                        **meta,
                        baseline=a,
                        added=b,
                        fraction=q,
                        selection=label,
                        n_blocks=int(m.sum()),
                        high_blocks=int(y[m].sum()),
                        median_ccn_cm3=float(np.median(c[m])) if m.any() else None,
                    )
                )


def main():
    assert not (OUT / "manifest.json").exists()
    mm = json.loads((MET / "manifest.json").read_text())
    om = json.loads((OLD / "manifest.json").read_text())
    for name, value in mm["outputs"].items():
        assert file_info(MET / name) == value, name
    for name in ("blocks.csv.gz", "monthly_threshold_support.csv"):
        assert file_info(OLD / name) == om["outputs"][name]
    d = ref.audited.read_site("SGP")
    assert (
        d["source_files"] == om["source_files"]
        and d["mapped_audit_file"] == om["mapped_audit_file"]
    )
    b = pd.read_csv(OLD / "blocks.csv.gz")
    for col in ["block_start", "block_end", "origin_1h", "origin_3h", "origin_6h"]:
        b[col] = pd.to_datetime(b[col], utc=True)
    ths = pd.read_csv(OLD / "monthly_threshold_support.csv").set_index("month")
    th = b.month.map(ths.threshold_cm3.where(ths.supported)).to_numpy()
    y = (b.mean_ccn_cm3.to_numpy() >= th).astype(int)
    h = pd.read_csv(MET / "hourly_primary.csv.gz", index_col=0)
    h.index = pd.to_datetime(h.index, utc=True)
    cut = pd.Timestamp(om["calibration_boundary"])
    outer = pd.Timestamp(om["outer_boundary"])
    rows = []
    gains = []
    budgets = []
    bg = []
    unique = []
    settings = []
    for lead in (1, 3, 6):
        good = b[f"origin_valid_{lead}h"].to_numpy(bool)
        ii = d["ts"].get_indexer(b.loc[good, f"origin_{lead}h"])
        assert (ii >= 0).all() and d["current_ok"][ii].all()
        for name, key in [("N15_300", "total"), ("f82", "f82")]:
            assert np.allclose(
                b.loc[good, f"{name}_{lead}h"], d[key][ii], rtol=1e-12, atol=1e-12
            )
        xs = ref.orig.block_features(b, lead)
        op = pd.read_csv(OLD / f"monthly_lead{lead}_predictions.csv.gz")
        assert (
            file_info(OLD / f"monthly_lead{lead}_predictions.csv.gz")
            == om["outputs"][f"monthly_lead{lead}_predictions.csv.gz"]
        )
        op = op.rename(
            columns={
                "p_calendar_total": "p_N",
                "p_calendar_total_f82": "p_N_f82",
                "p_calendar_N82": "p_N82",
            }
        )
        evaluate(
            op,
            dict(family="original_logistic", subset="original", lead_h=lead),
            ["N", "N_f82", "N82"],
            rows,
            gains,
            budgets,
            bg,
            unique,
        )
        aligned = h.reindex(pd.DatetimeIndex(b[f"origin_{lead}h"]))
        for rain in (False, True):
            subset = "primary_" + ("rain" if rain else "core")
            met = aligned[
                ["temperature", "rh", "pressure", "windspeed", "u", "v"]
            ].to_numpy()
            if rain:
                met = np.column_stack([met, np.log1p(aligned.rain_mm.to_numpy())])
            eligible = (
                (b.n_valid_ccn_hours >= 4)
                & good
                & np.isfinite(th)
                & np.isfinite(met).all(axis=1)
            )
            fit = eligible & (b.block_end <= cut)
            test = eligible & (b[f"origin_{lead}h"] >= outer)
            xsets = dict(
                N=xs["calendar_total"],
                N_f82=xs["calendar_total_f82"],
                N_met=np.column_stack([xs["calendar_total"], met]),
                N_met_f82=np.column_stack([xs["calendar_total_f82"], met]),
            )
            dates = pd.DatetimeIndex(
                b.loc[fit, f"origin_{lead}h"].dt.floor("D").unique()
            ).sort_values()
            inner = dates[int(0.8 * len(dates))]
            train = fit & (b.block_end <= inner)
            val = fit & (b[f"origin_{lead}h"] >= inner)
            assert (
                train.sum() > 100
                and val.sum() > 100
                and (len(np.unique(y[train])) == 2)
                and (len(np.unique(y[val])) == 2)
            )
            old = pd.read_csv(MET / f"{subset}_lead{lead}_predictions.csv.gz")
            assert np.array_equal(
                pd.to_datetime(old.block_start, utc=True), b.loc[test, "block_start"]
            )
            pred = old[[c for c in old if not c.startswith("p_")]].copy()
            for name, x in xsets.items():
                check = (
                    ref.orig.classifier()
                    .fit(x[fit], y[fit])
                    .predict_proba(x[test])[:, 1]
                )
                assert np.allclose(check, old["p_" + name], atol=1e-10, rtol=0)
                model = lgb.LGBMClassifier(**PARAM)
                model.fit(
                    x[train],
                    y[train],
                    eval_set=[(x[val], y[val])],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(25, verbose=False)],
                )
                best = int(model.best_iteration_)
                final = lgb.LGBMClassifier(**{**PARAM, "n_estimators": best}).fit(
                    x[fit], y[fit]
                )
                pred["p_" + name] = final.predict_proba(x[test])[:, 1]
                final.booster_.save_model(str(OUT / f"{subset}_lead{lead}_{name}.txt"))
                settings.append(
                    dict(
                        subset=subset,
                        lead_h=lead,
                        model=name,
                        n_fit=int(fit.sum()),
                        n_test=int(test.sum()),
                        inner_boundary=str(inner),
                        n_inner_train=int(train.sum()),
                        n_inner_validation=int(val.sum()),
                        best_iteration=best,
                    )
                )
            pred.to_csv(OUT / f"{subset}_lead{lead}_predictions.csv.gz", index=False)
            for family, p in [("logistic", old), ("lightgbm", pred)]:
                evaluate(
                    p,
                    dict(family=family, subset=subset, lead_h=lead),
                    list(xsets),
                    rows,
                    gains,
                    budgets,
                    bg,
                    unique,
                )
            print("DONE", subset, lead, flush=True)
    for name, records in [
        ("scores", rows),
        ("paired_gains", gains),
        ("selection_budgets", budgets),
        ("budget_gains", bg),
        ("distinct_selections", unique),
        ("fit_settings", settings),
    ]:
        pd.DataFrame(records).to_csv(OUT / (name + ".csv"), index=False)
    manifest = dict(
        status="complete",
        completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        lightgbm_version=lgb.__version__,
        lightgbm_settings=PARAM,
        budget_fractions=Q,
        logistic_reproduced=True,
        pnsd_gate_passed=True,
        source_manifest_file=file_info(MET / "manifest.json"),
        script_file=file_info(Path(__file__)),
        analysis_code_file=file_info(Path(__file__)),
        outputs={p.name: file_info(p) for p in OUT.iterdir() if p.is_file()},
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
