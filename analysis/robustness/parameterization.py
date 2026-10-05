"""Fixed ridge-penalty and fraction-parameterization sensitivity checks."""

from pathlib import Path
import os
from analysis.common.runtime import ROOT

os.umask(63)
from analysis.common.runtime import WORK

for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, average_precision_score
from analysis.diagnostics import size_fraction as previous
from analysis.preprocessing import complete_cohort as builder
from analysis.diagnostics import temporal as temporal
from analysis.models import score as scoring

for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
OUT = ROOT / "results/model_robustness"
SOURCE = ROOT / "results/complete_cohort"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, choices=previous.SITES)
    site = parser.parse_args().site
    OUT.mkdir(parents=True, exist_ok=True, mode=448)
    if (OUT / f"{site}_manifest.json").exists():
        raise FileExistsError(site)
    path = SOURCE / f"{site}_cohort.npz"
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    previous.SOURCE = SOURCE
    manifest, audit = previous.check_cohort(site, d)
    assert {
        p.name: previous.file_info(p) for p in builder.source_smps(site)
    } == manifest["source_csv_files"]
    assert (
        previous.file_info(previous.AUDIT_SOURCE / f"{site}_mapped24_audit.json")
        == manifest["mapped_audit_file"]
    )
    fit, test = (d["fit"].astype(bool), d["test"].astype(bool))
    bins = d["bins_now"].astype(float)
    total = bins.sum(axis=1)
    count, fraction = previous.fraction_above(
        bins, d["diameter_midpoint_nm"].astype(float), 82.0
    )
    fractions = bins / total[:, None]
    aux = d["aux"].astype(float)
    features = {
        "N": aux,
        "N_plus_f82": np.column_stack((aux, fraction)),
        "N82": np.column_stack((aux[:, :-1], np.log1p(count))),
        "full23": np.column_stack((aux, fractions[:, :-1])),
        "full24": np.column_stack((aux, fractions)),
    }
    y = d["y_ccn"].astype(float)
    timestamps = pd.to_datetime(d["timestamp"][test], utc=True)
    rows, gains, predictions = ([], [], [])
    frozen = pd.read_csv(
        ROOT / "results/complete_diagnostics/nine_site_model_scores.csv"
    )
    aliases = {
        "N": "total",
        "N_plus_f82": "total_plus_f82",
        "N82": "calibrated_Ngt82",
        "full23": "total_plus_24fractions",
    }
    for alpha in (1.0, 10.0, 100.0):
        preds = {}
        for name, x in features.items():
            keep = np.isfinite(x[fit]).any(axis=0)
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
            )
            p = model.fit(x[fit][:, keep], y[fit]).predict(x[test][:, keep])
            preds[name] = p
            value = previous.r2(y[test], p)
            if alpha == 10 and name in aliases:
                old = frozen[
                    (frozen.site == site)
                    & (frozen.task == "future_3h")
                    & (frozen.model == aliases[name])
                ]
                assert (
                    len(old) == 1 and abs(value - float(old.r2_log.iloc[0])) < 1e-08
                ), (site, name, value)
            rows.append(
                dict(
                    site=site,
                    alpha=alpha,
                    representation=name,
                    n_fit=int(fit.sum()),
                    n_test=int(test.sum()),
                    r2_log=value,
                    mae_cm3=float(np.abs(np.expm1(y[test]) - np.expm1(p)).mean()),
                )
            )
            predictions.append(
                pd.DataFrame(
                    dict(
                        site=site,
                        alpha=alpha,
                        representation=name,
                        timestamp=timestamps,
                        y_log=y[test],
                        prediction_log=p,
                    )
                )
            )
        for base, alt in [
            ("N", "N_plus_f82"),
            ("N_plus_f82", "full23"),
            ("full23", "full24"),
        ]:
            lo, hi = temporal.block_interval(
                y[test], preds[base], preds[alt], d["date"][test], 7, 270920
            )
            gains.append(
                dict(
                    site=site,
                    alpha=alpha,
                    base=base,
                    alternative=alt,
                    delta_r2=previous.r2(y[test], preds[alt])
                    - previous.r2(y[test], preds[base]),
                    lo95_7day=lo,
                    hi95_7day=hi,
                )
            )
    pd.DataFrame(rows).to_csv(OUT / f"{site}_ridge_scores.csv", index=False)
    pd.DataFrame(gains).to_csv(OUT / f"{site}_ridge_gains.csv", index=False)
    pd.concat(predictions).to_csv(
        OUT / f"{site}_ridge_predictions.csv.gz", index=False, compression="gzip"
    )
    if site == "SGP":
        matched = ROOT / "results/model_benchmark"
        contract = json.loads((matched / "common_cohort_manifest.json").read_text())
        assert (
            previous.file_info(matched / "common_cohort.npz") == contract["cohort_file"]
        )
        with np.load(matched / "common_cohort.npz") as z:
            m = {k: z[k] for k in z.files}
        train, cal, evaluate = (m[k].astype(bool) for k in ("train", "cal", "test"))
        allrows = []
        for context in ("current", "history6"):
            for n in (23, 24):
                seq = np.concatenate(
                    (m["x_N"][:, :, None], m["x_fraction24"][:, :, :n]), axis=2
                )
                particle = (
                    seq[:, -1, :] if context == "current" else seq.reshape(len(seq), -1)
                )
                x = np.column_stack((m["calendar"], particle)).astype("float32")
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=2000, random_state=270927),
                )
                model.fit(x[train], m["y"][train])
                pcal = model.predict_proba(x[cal])[:, 1]
                raw = model.predict_proba(x[evaluate])[:, 1]
                p, a, b = scoring.calibration(pcal, m["y"][cal], raw)
                hits, k = scoring.hits(m["y"][evaluate], p)
                allrows.append(
                    dict(
                        context=context,
                        fractions=n,
                        n_test=int(evaluate.sum()),
                        brier_raw=brier_score_loss(m["y"][evaluate], raw),
                        brier_calibrated=brier_score_loss(m["y"][evaluate], p),
                        average_precision=average_precision_score(m["y"][evaluate], p),
                        hits=hits,
                        selected=k,
                        calibration_intercept=a,
                        calibration_slope=b,
                    )
                )
        pd.DataFrame(allrows).to_csv(
            OUT / "SGP_logistic_parameterization.csv", index=False
        )
    (OUT / f"{site}_manifest.json").write_text(
        json.dumps(
            dict(
                status="completed",
                site=site,
                alpha10_original_scores_reproduced=True,
                preprocessing_gate_passed=True,
                cohort_file=previous.file_info(path),
                mapped_audit_file=manifest["mapped_audit_file"],
                script_file=previous.file_info(Path(__file__)),
                output_files={
                    p.name: previous.file_info(p) for p in OUT.glob(f"{site}_*.csv*")
                },
            ),
            indent=2,
        )
    )
    print(site, "completed", flush=True)


if __name__ == "__main__":
    main()
