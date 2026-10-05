"""Matched current/history tabular models for measured SGP high-CCN states."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

OUT = ROOT / "results/model_benchmark"
SEED = 270927
REPRESENTATIONS = ("N", "N82", "N_plus_f82", "full24")
FAMILIES = ("logistic", "random_forest", "lightgbm", "xgboost")


def features(d: dict[str, np.ndarray], name: str, context: str) -> np.ndarray:
    total = d["x_N"][:, :, None]
    count82 = d["x_N82"][:, :, None]
    fraction82 = d["x_f82"][:, :, None]
    if name == "N":
        sequence = total
    elif name == "N82":
        sequence = count82
    elif name == "N_plus_f82":
        sequence = np.concatenate((total, fraction82), axis=2)
    elif name == "full24":
        sequence = np.concatenate((total, d["x_fraction24"]), axis=2)
    else:
        raise ValueError(name)
    if context == "current":
        particle = sequence[:, -1, :]
    elif context == "history6":
        particle = sequence.reshape(len(sequence), -1)
    else:
        raise ValueError(context)
    return np.column_stack((d["calendar"], particle)).astype("float32")


def fit_model(
    family: str, x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray
):
    if family == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED),
        )
        model.fit(x[train], y[train])
        best = None
    elif family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=4,
            random_state=SEED,
        )
        model.fit(x[train], y[train])
        best = None
    elif family == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(
            x[train],
            y[train],
            eval_set=[(x[val], y[val])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        best = int(model.best_iteration_)
    elif family == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            early_stopping_rounds=25,
            random_state=SEED,
        )
        model.fit(x[train], y[train], eval_set=[(x[val], y[val])], verbose=False)
        best = int(model.best_iteration) + 1
    else:
        raise ValueError(family)
    return (model, best)


def main() -> None:
    dataset_path = OUT / "common_cohort.npz"
    manifest = json.loads((OUT / "common_cohort_manifest.json").read_text())
    if file_info(dataset_path) != manifest["cohort_file"]:
        raise ValueError("common cohort file changed")
    with np.load(dataset_path, allow_pickle=False) as file:
        d = {key: file[key] for key in file.files}
    y = d["y"].astype(int)
    train, val, cal, test = (
        d[key].astype(bool) for key in ("train", "val", "cal", "test")
    )
    results = []
    predictions = []
    background, _ = fit_model("logistic", d["calendar"], y, train, val)
    pval = background.predict_proba(d["calendar"][val])[:, 1]
    results.append(
        {
            "context": "current",
            "representation": "calendar",
            "model": "logistic",
            "seed": SEED,
            "n_train": int(train.sum()),
            "n_val": int(val.sum()),
            "n_cal": int(cal.sum()),
            "n_test": int(test.sum()),
            "best_iteration": None,
            "val_brier": float(brier_score_loss(y[val], pval)),
        }
    )
    for part, mask in (("cal", cal), ("test", test)):
        predictions.append(
            pd.DataFrame(
                {
                    "context": "current",
                    "representation": "calendar",
                    "model": "logistic",
                    "seed": SEED,
                    "partition": part,
                    "block_start": d["block_start"][mask],
                    "origin": d["origin"][mask],
                    "event": y[mask],
                    "p_raw": background.predict_proba(d["calendar"][mask])[:, 1],
                }
            )
        )
    for context in ("current", "history6"):
        for representation in REPRESENTATIONS:
            x = features(d, representation, context)
            if not np.isfinite(x).all():
                raise ValueError(f"nonfinite input: {context}, {representation}")
            for family in FAMILIES:
                model, best = fit_model(family, x, y, train, val)
                pval = model.predict_proba(x[val])[:, 1]
                pcal = model.predict_proba(x[cal])[:, 1]
                ptest = model.predict_proba(x[test])[:, 1]
                results.append(
                    {
                        "context": context,
                        "representation": representation,
                        "model": family,
                        "seed": SEED,
                        "n_train": int(train.sum()),
                        "n_val": int(val.sum()),
                        "n_cal": int(cal.sum()),
                        "n_test": int(test.sum()),
                        "best_iteration": best,
                        "val_brier": float(brier_score_loss(y[val], pval)),
                    }
                )
                for part, mask, p in (("cal", cal, pcal), ("test", test, ptest)):
                    predictions.append(
                        pd.DataFrame(
                            {
                                "context": context,
                                "representation": representation,
                                "model": family,
                                "seed": SEED,
                                "partition": part,
                                "block_start": d["block_start"][mask],
                                "origin": d["origin"][mask],
                                "event": y[mask],
                                "p_raw": p,
                            }
                        )
                    )
                print(
                    context,
                    representation,
                    family,
                    "val Brier",
                    round(results[-1]["val_brier"], 5),
                    flush=True,
                )
    pd.DataFrame(results).to_csv(OUT / "tabular_fit_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / "tabular_predictions.csv.gz", index=False, compression="gzip"
    )
    (OUT / "tabular_manifest.json").write_text(
        json.dumps(
            {
                "dataset_file": manifest["cohort_file"],
                "source_script_file": file_info(Path(__file__)),
                "contexts": ["current", "history6"],
                "representations": ["calendar"] + list(REPRESENTATIONS),
                "models": list(FAMILIES),
                "model_selection": "fixed hyperparameters; boosting early stopping on inner chronological validation only",
                "training": "common inner training subset; test not used for any fit or stopping",
                "seed": SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
