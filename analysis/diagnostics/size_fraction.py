"""Test how the fraction above 82 nm captures CCN size information.

Uses the audited, label-complete nine-site cohort. This is an
exploratory association/prediction analysis, not a causal effect estimator.
"""

from __future__ import annotations

from analysis.common.files import file_info
import argparse

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from analysis.common.runtime import ROOT

AUDIT_SOURCE = ROOT / "results/mapped"
SOURCE = ROOT / "results/complete_cohort"
OUT = ROOT / "results/complete_diagnostics"
SITES = ("ANX", "COR", "ENA", "GUC", "MAO", "MOS", "SBS_CP", "SBS_SPL", "SGP")
CUTOFFS_NM = (50.0, 65.0, 82.0, 100.0, 125.0, 150.0)
SEED = 260926


def fraction_above(
    bins: np.ndarray, diameter: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate above cutoff with fractional overlap of equal-log-width cells."""
    centers = np.log10(diameter.astype(float))
    edges = np.r_[
        centers[0] - (centers[1] - centers[0]) / 2,
        (centers[:-1] + centers[1:]) / 2,
        centers[-1] + (centers[-1] - centers[-2]) / 2,
    ]
    if not (np.all(np.diff(edges) > 0) and edges[0] < np.log10(cutoff) < edges[-1]):
        raise ValueError("invalid mapped bin edges or cutoff")
    weight = np.clip(
        (edges[1:] - np.maximum(edges[:-1], np.log10(cutoff))) / np.diff(edges),
        0.0,
        1.0,
    )
    count = bins.astype(float) @ weight
    total = bins.astype(float).sum(axis=1)
    return (count, count / total)


def regressor() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    den = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / den)


def paired_delta(
    y: np.ndarray,
    base: np.ndarray,
    added: np.ndarray,
    dates: np.ndarray,
    seed: int,
    n_boot: int = 1000,
) -> tuple[float, float, float]:
    """Date-block interval for the paired change in test R2, with no refitting."""
    unique, idx = np.unique(dates, return_inverse=True)
    n = np.bincount(idx)
    sy = np.bincount(idx, weights=y)
    sy2 = np.bincount(idx, weights=y * y)
    e0 = np.bincount(idx, weights=(y - base) ** 2)
    e1 = np.bincount(idx, weights=(y - added) ** 2)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    count = n[draw].sum(axis=1)
    ysum = sy[draw].sum(axis=1)
    sst = sy2[draw].sum(axis=1) - ysum * ysum / count
    gains = (e0[draw].sum(axis=1) - e1[draw].sum(axis=1)) / sst
    return (
        r2(y, added) - r2(y, base),
        *np.quantile(gains[np.isfinite(gains)], [0.025, 0.975]),
    )


def check_cohort(site: str, d: dict[str, np.ndarray]) -> tuple[dict, dict]:
    manifest_file = SOURCE / f"{site}_cohort_manifest.json"
    audit_file = AUDIT_SOURCE / f"{site}_mapped24_audit.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    if not (
        manifest.get("new_preflight_passed")
        and audit.get("preprocessing_gate_passed")
        and audit.get("conservation_gate_passed")
        and audit.get("supplied_total_gate_passed")
    ):
        raise ValueError(f"{site}: PNSD audit gate is not passed")
    b = d["bins_now"].astype(float)
    if (
        b.ndim != 2
        or b.shape[1] != 24
        or (not np.isfinite(b).all())
        or (b < 0).any()
        or (b.sum(axis=1) <= 0).any()
    ):
        raise ValueError(f"{site}: 24-bin concentration invalid")
    if not (np.isfinite(d["y_ccn"]).all() and np.isfinite(d["current_ccn"]).all()):
        raise ValueError(f"{site}: measured CCN labels invalid")
    fit, val, test = (d[key].astype(bool) for key in ("fit", "val", "test"))
    if np.any(fit & val) or np.any(fit & test) or np.any(val & test):
        raise ValueError(f"{site}: split overlap")
    if int(test.sum()) != manifest["n_test"] or int(fit.sum()) != manifest["n_fit"]:
        raise ValueError(f"{site}: cohort counts differ from manifest")
    if len(d["timestamp"]) != len(b) or len(np.unique(d["timestamp"])) != len(b):
        raise ValueError(f"{site}: timestamp mismatch or duplicate")
    if max(d["timestamp"][fit]) >= min(d["timestamp"][test]):
        raise ValueError(f"{site}: fit/test time order invalid")
    return (manifest, audit)


def analyze_site(
    site: str, output: Path
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict, dict]:
    cohort_file = SOURCE / f"{site}_cohort.npz"
    with np.load(cohort_file) as z:
        d = {key: z[key] for key in z.files}
    manifest, audit = check_cohort(site, d)
    fit, test = (d["fit"].astype(bool), d["test"].astype(bool))
    bins = d["bins_now"].astype(float)
    diameter = d["diameter_midpoint_nm"].astype(float)
    total = bins.sum(axis=1)
    count82, frac82 = fraction_above(bins, diameter, 82.0)
    future_count82, _ = fraction_above(d["bins_future"].astype(float), diameter, 82.0)
    fractions = bins / total[:, None]
    aux = d["aux"].astype(float)
    if not np.allclose(aux[:, -1], np.log1p(total), atol=1e-05):
        raise ValueError(f"{site}: aux total and 24-bin total differ")
    features = {
        "total": aux,
        "total_plus_f82": np.column_stack((aux, frac82)),
        "total_plus_24fractions": np.column_stack((aux, fractions[:, :-1])),
        "Ngt82_only": np.log1p(count82)[:, None],
        "calibrated_Ngt82": np.column_stack((aux[:, :-1], np.log1p(count82))),
        "current_ccn": np.column_stack((aux, d["current_ccn"])),
        "current_ccn_plus_f82": np.column_stack((aux, d["current_ccn"], frac82)),
        "current_ccn_plus_log_Ngt82": np.column_stack(
            (aux, d["current_ccn"], np.log1p(count82))
        ),
        "current_ccn_plus_24fractions": np.column_stack(
            (aux, d["current_ccn"], fractions[:, :-1])
        ),
    }
    tasks = {
        "concurrent": d["current_ccn"].astype(float),
        "future_3h": d["y_ccn"].astype(float),
    }
    score_rows, delta_rows, scan_rows, slope_rows = ([], [], [], [])
    for task, y in tasks.items():
        preds = {"Ngt82_direct": np.log1p(count82[test])}
        score_rows.append(
            {
                "site": site,
                "task": task,
                "model": "Ngt82_direct",
                "n_fit": int(fit.sum()),
                "n_test": int(test.sum()),
                "n_test_dates": int(len(np.unique(d["date"][test]))),
                "r2_log": r2(y[test], preds["Ngt82_direct"]),
                "mae_log": float(np.mean(np.abs(y[test] - preds["Ngt82_direct"]))),
            }
        )
        for name, x in features.items():
            if task == "concurrent" and name.startswith("current_ccn"):
                continue
            keep = np.isfinite(x[fit]).any(axis=0)
            model = regressor().fit(x[fit][:, keep], y[fit])
            preds[name] = model.predict(x[test][:, keep])
            if name == "total_plus_f82":
                q25, q75 = np.quantile(frac82[fit], [0.25, 0.75])
                beta = float(
                    model.named_steps["ridge"].coef_[-1]
                    / model.named_steps["scale"].scale_[-1]
                )
                slope_rows.append(
                    {
                        "site": site,
                        "task": task,
                        "f82_fit_q25": float(q25),
                        "f82_fit_q75": float(q75),
                        "partial_slope_log1p_ccn_per_unit_f82": beta,
                        "model_ratio_1p_ccn_q75_vs_q25_at_fixed_covariates": float(
                            np.exp(beta * (q75 - q25))
                        ),
                        "interpretation": "trained-model conditional association, not a causal effect",
                    }
                )
            score_rows.append(
                {
                    "site": site,
                    "task": task,
                    "model": name,
                    "n_fit": int(fit.sum()),
                    "n_test": int(test.sum()),
                    "n_test_dates": int(len(np.unique(d["date"][test]))),
                    "r2_log": r2(y[test], preds[name]),
                    "mae_log": float(np.mean(np.abs(y[test] - preds[name]))),
                }
            )
        comparisons = [
            ("total", "total_plus_f82"),
            ("total", "total_plus_24fractions"),
            ("total_plus_f82", "total_plus_24fractions"),
            ("total", "calibrated_Ngt82"),
            ("calibrated_Ngt82", "total_plus_24fractions"),
            ("Ngt82_direct", "Ngt82_only"),
        ]
        if task == "future_3h":
            comparisons += [
                ("current_ccn", "current_ccn_plus_f82"),
                ("current_ccn", "current_ccn_plus_log_Ngt82"),
                ("current_ccn", "current_ccn_plus_24fractions"),
            ]
        for i, (base, added) in enumerate(comparisons):
            estimate, lo, hi = paired_delta(
                y[test], preds[base], preds[added], d["date"][test], SEED + i
            )
            delta_rows.append(
                {
                    "site": site,
                    "task": task,
                    "base": base,
                    "added": added,
                    "delta_r2": estimate,
                    "lo95": lo,
                    "hi95": hi,
                    "n_test": int(test.sum()),
                    "n_test_dates": int(len(np.unique(d["date"][test]))),
                }
            )
        for cutoff in CUTOFFS_NM:
            _, fraction = fraction_above(bins, diameter, cutoff)
            x = np.column_stack((aux, fraction))
            keep = np.isfinite(x[fit]).any(axis=0)
            pred = regressor().fit(x[fit][:, keep], y[fit]).predict(x[test][:, keep])
            scan_rows.append(
                {
                    "site": site,
                    "task": task,
                    "cutoff_nm": cutoff,
                    "delta_r2_vs_total": r2(y[test], pred)
                    - r2(y[test], preds["total"]),
                }
            )
        if "current_closure_ok" in d and "future_closure_ok" in d:
            rows = pd.DataFrame(
                {
                    "site": site,
                    "task": task,
                    "date": d["date"][test],
                    "timestamp": d["timestamp"][test],
                    "target_log": y[test],
                    "closure_pair_ok": d["current_closure_ok"][test].astype(bool)
                    & d["future_closure_ok"][test].astype(bool),
                }
            )
            for name, pred in preds.items():
                rows[f"pred_{name}"] = pred
            rows.to_csv(output / f"{site}_{task}_test_predictions.csv.gz", index=False)
    info = {
        "site": site,
        "cohort_file": file_info(cohort_file),
        "source_manifest_file": file_info(SOURCE / f"{site}_cohort_manifest.json"),
        "mapped_audit_file": file_info(AUDIT_SOURCE / f"{site}_mapped24_audit.json"),
        "mapped_over_native_median": audit["conservation_mapped_over_native"]["median"],
        "mapped_over_frozen_total_median": manifest[
            "mapped_total_over_frozen_total_q05_median_q95"
        ][1],
        "n_fit": int(fit.sum()),
        "n_val_not_used": int(d["val"].sum()),
        "n_test": int(test.sum()),
    }
    current = np.expm1(d["current_ccn"][test].astype(float))
    future = np.expm1(d["y_ccn"][test].astype(float))
    now_pool = count82[test]
    future_pool = future_count82[test]
    bridge = {
        "site": site,
        "n_test": int(test.sum()),
        "ccn_now_over_Ngt82_now_q10": float(np.quantile(current / now_pool, 0.1)),
        "ccn_now_over_Ngt82_now_median": float(np.median(current / now_pool)),
        "ccn_now_over_Ngt82_now_q90": float(np.quantile(current / now_pool, 0.9)),
        "ccn_future_over_Ngt82_future_median": float(np.median(future / future_pool)),
        "r2_log_Ngt82_pool_persistence": r2(np.log1p(future_pool), np.log1p(now_pool)),
        "r2_log_future_ccn_from_future_Ngt82_lookahead": r2(
            np.log1p(future), np.log1p(future_pool)
        ),
        "r2_log_future_ccn_from_now_Ngt82": r2(np.log1p(future), np.log1p(now_pool)),
        "median_abs_log_Ngt82_3h_change": float(
            np.median(np.abs(np.log1p(future_pool) - np.log1p(now_pool)))
        ),
    }
    return (score_rows, delta_rows, scan_rows, slope_rows, info, bridge)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cohort-source", type=Path, default=ROOT / "results/complete_cohort"
    )
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    global SOURCE
    SOURCE = args.cohort_source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    scores, deltas, scans, slopes, sites, bridges = ([], [], [], [], [], [])
    for site in SITES:
        a, b, c, e, info, bridge = analyze_site(site, args.output)
        scores.extend(a)
        deltas.extend(b)
        scans.extend(c)
        slopes.extend(e)
        sites.append(info)
        bridges.append(bridge)
        print(site, "fit", info["n_fit"], "test", info["n_test"], flush=True)
    pd.DataFrame(scores).to_csv(args.output / "nine_site_model_scores.csv", index=False)
    pd.DataFrame(deltas).to_csv(
        args.output / "nine_site_paired_deltas.csv", index=False
    )
    pd.DataFrame(scans).to_csv(args.output / "nine_site_diameter_scan.csv", index=False)
    pd.DataFrame(slopes).to_csv(
        args.output / "nine_site_fraction_slopes.csv", index=False
    )
    pd.DataFrame(bridges).to_csv(
        args.output / "nine_site_threshold_bridge.csv", index=False
    )
    strata_rows = []
    if "current_closure_ok" in np.load(SOURCE / f"{SITES[0]}_cohort.npz").files:
        for site in SITES:
            for task in ("concurrent", "future_3h"):
                frame = pd.read_csv(
                    args.output / f"{site}_{task}_test_predictions.csv.gz"
                )
                for stratum, part in (
                    ("closure_pass", frame[frame.closure_pair_ok]),
                    ("closure_fail_either", frame[~frame.closure_pair_ok]),
                ):
                    if len(part) < 2 or part.target_log.var() <= 0:
                        continue
                    y = part.target_log.to_numpy(float)
                    base = r2(y, part.pred_total.to_numpy(float))
                    shape = r2(y, part.pred_total_plus_f82.to_numpy(float))
                    direct = r2(y, part.pred_Ngt82_direct.to_numpy(float))
                    strata_rows.append(
                        {
                            "site": site,
                            "task": task,
                            "stratum": stratum,
                            "n_test": int(len(part)),
                            "n_dates": int(part.date.nunique()),
                            "r2_total": base,
                            "r2_total_plus_f82": shape,
                            "delta_r2": shape - base,
                            "r2_Ngt82_direct": direct,
                        }
                    )
        pd.DataFrame(strata_rows).to_csv(
            args.output / "closure_strata_sensitivity.csv", index=False
        )
    cohort_contract = json.loads(
        (SOURCE / f"{SITES[0]}_cohort_manifest.json").read_text(encoding="utf-8")
    )
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "analysis_code": Path(__file__).name,
                "cohort_source": str(SOURCE.relative_to(ROOT)),
                "target": "measured CCN at 0.4% SS, concurrent and exact +3 h",
                "cohort": cohort_contract.get(
                    "cohort", "audited 24-bin CCN/PNSD cohort"
                ),
                "threshold_nm": 82.0,
                "threshold_source": "Zabala et al. ACP 2026, DOI 10.5194/acp-26-3697-2026",
                "threshold_integration": "partial target-bin overlap in log10 diameter; uniform density within each mapped bin",
                "fit": "source cohort fit rows only; chronological test; fixed Ridge alpha=10; see site cohort manifests for exact split",
                "bootstrap": "1000 resamples of test calendar dates, predictions held fixed",
                "diameter_scan_nm": CUTOFFS_NM,
                "interpretation": "conditional prediction association, not a do-intervention or growth flux",
                "sites": sites,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = (
        pd.DataFrame(deltas)
        .groupby(["task", "base", "added"], sort=False)
        .agg(
            median_delta_r2=("delta_r2", "median"),
            positive_sites=("delta_r2", lambda x: int((x > 0).sum())),
            lower_ci_positive_sites=("lo95", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
