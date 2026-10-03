"""Score same-cohort high-CCN models on physical information, not leaderboards."""

from __future__ import annotations
from pathlib import Path
import os
from runtime import ROOT
from runtime import WORK

os.umask(63)
WORK.mkdir(parents=True, exist_ok=True, mode=448)
for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[key] = str(WORK)
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss

OUT = ROOT / "results/model_benchmark"
SEED = 270929
REPRESENTATIONS = ("N", "N82", "N_plus_f82", "full24")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration(
    pcal: np.ndarray, ycal: np.ndarray, ptest: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Monotone Platt map using calibration period only."""
    zcal = logit(np.clip(pcal, 1e-06, 1 - 1e-06))
    ztest = logit(np.clip(ptest, 1e-06, 1 - 1e-06))

    def loss(ab: np.ndarray) -> float:
        q = expit(ab[0] + ab[1] * zcal)
        q = np.clip(q, 1e-09, 1 - 1e-09)
        return float(-np.mean(ycal * np.log(q) + (1 - ycal) * np.log1p(-q)))

    fit = minimize(
        loss,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=((None, None), (0.0, None)),
    )
    if not fit.success:
        raise ValueError(f"probability calibration failed: {fit.message}")
    a, b = map(float, fit.x)
    return (expit(a + b * ztest), a, b)


def hits(y: np.ndarray, p: np.ndarray, fraction: float = 0.1) -> tuple[int, int]:
    k = math.ceil(fraction * len(y))
    index = np.argsort(-p, kind="stable")[:k]
    return (int(y[index].sum()), k)


def paired_intervals(
    y: np.ndarray,
    pbase: np.ndarray,
    palt: np.ndarray,
    dates: pd.DatetimeIndex,
    seed: int,
    draws: int = 1000,
    pbase_cal: np.ndarray | None = None,
    palt_cal: np.ndarray | None = None,
) -> dict:
    """Fixed-model paired seven-day block intervals for loss and top-10% precision."""
    weeks = dates.asi8 // 86400000000000 // 7
    unique = np.unique(weeks)
    positions = [np.flatnonzero(weeks == w) for w in unique]
    rng = np.random.default_rng(seed)
    brier, precision, calibrated_brier = ([], [], [])
    for _ in range(draws):
        sampled = rng.integers(0, len(unique), size=len(unique))
        ix = np.concatenate([positions[j] for j in sampled])
        yy, aa, bb = (y[ix], pbase[ix], palt[ix])
        brier.append(float(np.mean((yy - aa) ** 2 - (yy - bb) ** 2)))
        if pbase_cal is not None and palt_cal is not None:
            calibrated_brier.append(
                float(np.mean((yy - pbase_cal[ix]) ** 2 - (yy - palt_cal[ix]) ** 2))
            )
        h0, k = hits(yy, aa)
        h1, _ = hits(yy, bb)
        precision.append((h1 - h0) / k)
    blo, bhi = np.quantile(brier, [0.025, 0.975])
    plo, phi = np.quantile(precision, [0.025, 0.975])
    output = {
        "brier_gain_lo95_7day": float(blo),
        "brier_gain_hi95_7day": float(bhi),
        "precision_gain_lo95_7day": float(plo),
        "precision_gain_hi95_7day": float(phi),
        "n_seven_day_blocks": int(len(unique)),
    }
    if calibrated_brier:
        clo, chi = np.quantile(calibrated_brier, [0.025, 0.975])
        output["calibrated_brier_gain_lo95_7day"] = float(clo)
        output["calibrated_brier_gain_hi95_7day"] = float(chi)
    return output


def load_groups() -> tuple[dict, dict]:
    dataset = OUT / "common_cohort.npz"
    manifest = json.loads((OUT / "common_cohort_manifest.json").read_text())
    if digest(dataset) != manifest["common_cohort_sha256"]:
        raise ValueError("common cohort hash changed")
    with np.load(dataset, allow_pickle=False) as file:
        d = {key: file[key] for key in file.files}
    expected = {
        part: {
            "time": d["block_start"][d[part].astype(bool)],
            "event": d["y"][d[part].astype(bool)].astype(int),
        }
        for part in ("cal", "test")
    }
    tables = []
    for name in ("tabular", "neural"):
        path = OUT / f"{name}_predictions.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        tables.append(pd.read_csv(path))
    frame = pd.concat(tables, ignore_index=True)
    if frame.duplicated(
        ["context", "representation", "model", "seed", "partition", "block_start"]
    ).any():
        raise ValueError("duplicate model predictions")
    groups = {}
    for key, part in frame.groupby(
        ["context", "representation", "model", "seed"], sort=True
    ):
        data = {}
        for partition in ("cal", "test"):
            rows = part.loc[part.partition == partition].sort_values("block_start")
            if not np.array_equal(
                rows.block_start.to_numpy(str), expected[partition]["time"]
            ):
                raise ValueError(f"unequal evaluation rows: {key}, {partition}")
            if not np.array_equal(
                rows.event.to_numpy(int), expected[partition]["event"]
            ):
                raise ValueError(f"changed target labels: {key}, {partition}")
            p = rows.p_raw.to_numpy(float)
            if not np.isfinite(p).all() or np.min(p) < 0 or np.max(p) > 1:
                raise ValueError(f"invalid probabilities: {key}, {partition}")
            data[partition] = p
        groups[key] = data
    expected_tabular = 1 + 4 * 4 * 2
    expected_neural = 2 * 4 * 2
    if len(groups) != expected_tabular + expected_neural:
        raise ValueError(f"missing model-input runs: {len(groups)}")
    return (groups, d)


def main() -> None:
    groups, d = load_groups()
    test = d["test"].astype(bool)
    cal = d["cal"].astype(bool)
    y = d["y"][test].astype(int)
    ycal = d["y"][cal].astype(int)
    dates = pd.DatetimeIndex(pd.to_datetime(d["block_start"][test], utc=True))
    scores = []
    calibrated_groups = {}
    for (context, representation, model, seed), p in groups.items():
        raw = p["test"]
        calibrated, intercept, slope = calibration(p["cal"], ycal, raw)
        calibrated_groups[context, representation, model, seed] = calibrated
        hit, k = hits(y, raw)
        scores.append(
            {
                "context": context,
                "representation": representation,
                "model": model,
                "seed": seed,
                "n_test": int(len(y)),
                "events": int(y.sum()),
                "selected_10pct": k,
                "hits_10pct": hit,
                "precision_10pct": hit / k,
                "recall_10pct": hit / y.sum(),
                "brier_raw": float(brier_score_loss(y, raw)),
                "brier_calibrated": float(brier_score_loss(y, calibrated)),
                "ap": float(average_precision_score(y, raw)),
                "cal_intercept": intercept,
                "cal_slope_nonnegative": slope,
            }
        )
    score = pd.DataFrame(scores).sort_values(
        ["context", "model", "seed", "representation"]
    )
    score.to_csv(OUT / "matched_model_scores.csv", index=False)
    comparisons = []
    count = 0
    by_family = {}
    for key, data in groups.items():
        context, representation, model, seed = key
        by_family.setdefault((context, model, seed), {})[representation] = data["test"]
    for (context, model, seed), variants in sorted(by_family.items()):
        for base, alternative in (
            ("N", "N82"),
            ("N", "N_plus_f82"),
            ("N", "full24"),
            ("N82", "full24"),
        ):
            p0, p1 = (variants[base], variants[alternative])
            q0 = calibrated_groups[context, base, model, seed]
            q1 = calibrated_groups[context, alternative, model, seed]
            h0, k = hits(y, p0)
            h1, _ = hits(y, p1)
            intervals = paired_intervals(
                y, p0, p1, dates, SEED + count, pbase_cal=q0, palt_cal=q1
            )
            comparisons.append(
                {
                    "context": context,
                    "model": model,
                    "seed": seed,
                    "base": base,
                    "alternative": alternative,
                    "n_test": int(len(y)),
                    "events": int(y.sum()),
                    "brier_gain": float(np.mean((y - p0) ** 2 - (y - p1) ** 2)),
                    "calibrated_brier_gain": float(
                        np.mean((y - q0) ** 2 - (y - q1) ** 2)
                    ),
                    "ap_gain": float(
                        average_precision_score(y, p1) - average_precision_score(y, p0)
                    ),
                    "selected_10pct": k,
                    "hits_base": h0,
                    "hits_alternative": h1,
                    "additional_true_states": h1 - h0,
                    "precision_gain_10pct": (h1 - h0) / k,
                    **intervals,
                }
            )
            count += 1
    pd.DataFrame(comparisons).to_csv(
        OUT / "paired_representation_gains.csv", index=False
    )
    history = []
    for model in ("logistic", "random_forest", "lightgbm", "xgboost"):
        for representation in REPRESENTATIONS:
            p0 = groups["current", representation, model, 270927]["test"]
            p1 = groups["history6", representation, model, 270927]["test"]
            h0, k = hits(y, p0)
            h1, _ = hits(y, p1)
            history.append(
                {
                    "model": model,
                    "representation": representation,
                    "brier_gain_history_vs_current": float(
                        np.mean((y - p0) ** 2 - (y - p1) ** 2)
                    ),
                    "additional_true_states_history": h1 - h0,
                    "current_hits_10pct": h0,
                    "history_hits_10pct": h1,
                    **paired_intervals(y, p0, p1, dates, SEED + count),
                }
            )
            count += 1
    pd.DataFrame(history).to_csv(OUT / "paired_history_gains.csv", index=False)
    context = pd.DataFrame(
        {
            "block_start": dates,
            "season": np.asarray(("DJF", "MAM", "JJA", "SON"), dtype=object)[
                dates.month % 12 // 3
            ],
            "year": dates.year,
            "event": y,
        }
    )
    strata = []
    for (ctx, representation, model, seed), p in groups.items():
        pred = p["test"]
        for group in ("season", "year"):
            for value, indices in context.groupby(group).indices.items():
                yy, pp = (y[indices], pred[indices])
                strata.append(
                    {
                        "context": ctx,
                        "representation": representation,
                        "model": model,
                        "seed": seed,
                        "stratum_type": group,
                        "stratum": str(value),
                        "n_test": int(len(indices)),
                        "events": int(yy.sum()),
                        "brier_raw": float(np.mean((yy - pp) ** 2)),
                        "ap": (
                            float(average_precision_score(yy, pp))
                            if yy.sum()
                            else np.nan
                        ),
                    }
                )
    pd.DataFrame(strata).to_csv(OUT / "model_season_year_scores.csv", index=False)
    report = {
        "status": "scored",
        "dataset_sha256": digest(OUT / "common_cohort.npz"),
        "tabular_sha256": digest(OUT / "tabular_predictions.csv.gz"),
        "neural_sha256": digest(OUT / "neural_predictions.csv.gz"),
        "n_models": int(len(score)),
        "n_paired_comparisons": int(len(comparisons)),
        "n_history_comparisons": int(len(history)),
        "n_test": int(len(y)),
        "events": int(y.sum()),
        "selection": "retrospective 10% of identical held-out six-hour blocks; not real-time alarm",
        "uncertainty": "1000 fixed-model paired seven-day block resamples",
        "calibration": "monotone Platt fitted on prior calibration period only; ranking evaluated on raw probabilities",
        "test_already_opened": True,
    }
    (OUT / "scoring_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
