"""Review-driven same-endpoint controls and transparent SGP block reporting."""

from pathlib import Path
import os
from runtime import ROOT
from runtime import WORK

os.umask(63)
WORK.mkdir(parents=True, exist_ok=True, mode=448)
for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
import primary_environmental as original
import shared_temporal as audited

for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
OUT = ROOT / "results/block_robustness"
OLD = original.OUT
SEED = 270920


def sha(path):
    return original.digest(path)


def rank_selection(p):
    k = math.ceil(0.1 * len(p))
    return np.argsort(-p, kind="stable")[:k]


def score(y, p):
    chosen = rank_selection(p)
    hits = int(y[chosen].sum())
    return dict(
        n_test=len(y),
        high_blocks=int(y.sum()),
        brier=float(brier_score_loss(y, p)),
        average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        selected=len(chosen),
        hits=hits,
        nonhigh_selected=len(chosen) - hits,
        missed_high=int(y.sum()) - hits,
        precision=hits / len(chosen),
        recall=hits / int(y.sum()) if y.sum() else None,
        distinct_probabilities=len(np.unique(p)),
    )


def interval(y, p0, p1, ts, seed):
    weeks = pd.DatetimeIndex(ts).asi8 // (7 * 86400000000000)
    groups = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    rng = np.random.default_rng(seed)
    brier, precision = ([], [])
    for _ in range(1000):
        ix = np.concatenate(
            [groups[j] for j in rng.integers(len(groups), size=len(groups))]
        )
        z = y[ix]
        a, b = (p0[ix], p1[ix])
        brier.append(float(np.mean((z - a) ** 2 - (z - b) ** 2)))
        precision.append(
            float(z[rank_selection(b)].mean() - z[rank_selection(a)].mean())
        )
    return dict(
        brier_gain_lo95=float(np.quantile(brier, 0.025)),
        brier_gain_hi95=float(np.quantile(brier, 0.975)),
        precision_gain_lo95=float(np.quantile(precision, 0.025)),
        precision_gain_hi95=float(np.quantile(precision, 0.975)),
        bootstrap_draws=1000,
        n_seven_day_blocks=len(groups),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True, mode=448)
    if (OUT / "manifest.json").exists():
        raise FileExistsError(
            "Completed V20 block outputs already exist; do not overwrite"
        )
    prior_manifest = json.loads((OLD / "manifest.json").read_text())
    for name, digest in prior_manifest["target_source_sha256"].items():
        assert sha(original.builder.SOURCE / "ccn" / name) == digest, name
    d = audited.read_site("SGP")
    blocks, _ = original.sgp_blocks(d)
    start = pd.DatetimeIndex(blocks.block_start)
    end = pd.DatetimeIndex(blocks.block_end)
    cal_cut = pd.Timestamp(prior_manifest["calibration_cut"])
    outer = pd.Timestamp(prior_manifest["outer_test_cut"])
    assert outer == d["cut"]
    monthly = pd.read_csv(OLD / "sgp_fit_month_thresholds.csv").set_index("month")
    threshold = blocks.month.map(monthly.high_threshold_cm3).to_numpy(float)
    y_mean = (blocks.mean_ccn_cm3.to_numpy(float) >= threshold).astype(int)
    blocks["mean_high_threshold_cm3"] = threshold
    blocks["mean_high"] = y_mean
    ccn = original.builder.load_ccn_instrument_only()["SGP"]
    raw_y = pd.to_numeric(ccn.N_CCN_mean_STP_B, errors="coerce")
    good = np.isfinite(raw_y) & (raw_y >= 0) & ccn.ccn_instrument_ok.astype(bool)
    hourly = pd.Series(
        raw_y.loc[good].to_numpy(float),
        index=pd.DatetimeIndex(ccn.loc[good, "timestamp"]),
    )
    assert not hourly.index.has_duplicates
    values = np.column_stack(
        [
            hourly.reindex(start + pd.Timedelta(hours=k)).to_numpy(float)
            for k in range(6)
        ]
    )
    assert np.array_equal(np.isfinite(values).sum(axis=1), blocks.n_valid_ccn_hours)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        block_median = np.nanmedian(values, axis=1)
        block_max = np.nanmax(values, axis=1)
    blocks["median_ccn_cm3"] = block_median
    blocks["max_ccn_cm3"] = block_max
    blocks["hours_ge_mean_threshold"] = (values >= threshold[:, None]).sum(axis=1)
    blocks["max_hour_share_of_sum"] = np.divide(
        block_max,
        np.nansum(values, axis=1),
        out=np.full(len(blocks), np.nan),
        where=np.nansum(values, axis=1) > 0,
    )
    base_eligible = blocks.n_valid_ccn_hours.to_numpy() >= 4
    common = base_eligible.copy()
    for lead in (1, 3, 6):
        common &= blocks[f"origin_valid_{lead}h"].to_numpy(bool)
    threshold_fit = common & (end <= cal_cut)
    med_table = blocks.loc[threshold_fit].groupby("month").median_ccn_cm3.quantile(0.75)
    median_threshold = blocks.month.map(med_table).to_numpy(float)
    y_median = (block_median >= median_threshold).astype(int)
    fit_hourly = hourly[hourly.index < cal_cut]
    hourly_q75 = fit_hourly.groupby(fit_hourly.index.month).quantile(0.75)
    sustained_threshold = blocks.month.map(hourly_q75).to_numpy(float)
    y_sustained = np.all(values >= sustained_threshold[:, None], axis=1).astype(int)
    blocks["median_high_threshold_cm3"] = median_threshold
    blocks["median_high"] = y_median
    blocks["sustained_hourly_threshold_cm3"] = sustained_threshold
    blocks["sustained_high"] = y_sustained
    stages, scores, gains, calibration, coverage, purge_rows = ([], [], [], [], [], [])
    cases = {}

    def run_case(case, labels, eligible, features, lead=3, shared=False, pairs=None):
        eligible = np.asarray(eligible, bool)
        origin = pd.DatetimeIndex(blocks[f"origin_{lead}h"])
        fit = eligible & (end <= cal_cut)
        cal = eligible & (start >= cal_cut) & (end <= outer)
        test = eligible & (origin >= outer)
        if shared:
            test &= pd.DatetimeIndex(blocks.origin_6h) >= outer
        assert (
            not (fit & cal).any()
            and (not (cal & test).any())
            and (not (fit & test).any())
        )
        assert (end[fit] <= cal_cut).all() and (end[cal] <= outer).all()
        assert (origin[test] >= outer).all()
        if len(np.unique(labels[fit])) != 2:
            raise ValueError(f"{case}: fitting classes insufficient; report explicitly")
        for stage, mask in [("fit", fit), ("calibration", cal), ("test", test)]:
            stages.append(
                dict(
                    case=case,
                    stage=stage,
                    lead_h=lead,
                    n_blocks=int(mask.sum()),
                    high_blocks=int(labels[mask].sum()),
                    block_start_min=str(start[mask].min()),
                    block_start_max=str(start[mask].max()),
                    block_end_exclusive_max=str(end[mask].max()),
                    origin_min=str(origin[mask].min()),
                    origin_max=str(origin[mask].max()),
                    calibration_boundary=str(cal_cut),
                    test_boundary=str(outer),
                )
            )
            for month in range(1, 13):
                m = mask & (blocks.month.to_numpy() == month)
                coverage.append(
                    dict(
                        case=case,
                        stage=stage,
                        month=month,
                        n_blocks=int(m.sum()),
                        high_blocks=int(labels[m].sum()),
                    )
                )
        for boundary_name, boundary in [
            ("fit_calibration", cal_cut),
            ("calibration_test", outer),
        ]:
            crossing = eligible & (origin < boundary) & (end > boundary)
            for row in np.flatnonzero(crossing):
                purge_rows.append(
                    dict(
                        case=case,
                        boundary_name=boundary_name,
                        boundary=str(boundary),
                        origin=str(origin[row]),
                        block_start=str(start[row]),
                        block_end_exclusive=str(end[row]),
                        included_fit=bool(fit[row]),
                        included_calibration=bool(cal[row]),
                        included_test=bool(test[row]),
                    )
                )
        y = labels[test]
        prior = float(labels[fit].mean())
        prediction = blocks.loc[
            test,
            [
                "block_start",
                "block_end",
                "month",
                "season",
                "n_valid_ccn_hours",
                "mean_ccn_cm3",
                "median_ccn_cm3",
                "max_ccn_cm3",
                "hours_ge_mean_threshold",
                "max_hour_share_of_sum",
            ],
        ].copy()
        prediction["origin"] = origin[test]
        prediction["event"] = y
        prediction["p_fitting_frequency"] = prior
        for model_name, x in features.items():
            fitted = original.classifier().fit(x[fit], labels[fit])
            prediction["p_" + model_name] = fitted.predict_proba(x[test])[:, 1]
        preds = {
            col[2:]: prediction[col].to_numpy(float)
            for col in prediction
            if col.startswith("p_")
        }
        for name, p in preds.items():
            result = dict(
                case=case,
                lead_h=lead,
                model=name,
                n_fit=int(fit.sum()),
                n_calibration=int(cal.sum()),
                fitting_frequency=prior,
                **score(y, p),
            )
            result["brier_skill_vs_fit_frequency"] = 1 - result[
                "brier"
            ] / brier_score_loss(y, preds["fitting_frequency"])
            scores.append(result)
            bins = np.minimum((p * 10).astype(int), 9)
            for k in range(10):
                m = bins == k
                calibration.append(
                    dict(
                        case=case,
                        model=name,
                        bin_lower=k / 10,
                        bin_upper=(k + 1) / 10,
                        n=int(m.sum()),
                        mean_probability=float(p[m].mean()) if m.any() else None,
                        observed_frequency=float(y[m].mean()) if m.any() else None,
                    )
                )
        comparisons = pairs or [
            ("calendar_total", "calendar_total_f82"),
            ("calendar", "calendar_total_f82"),
            ("fitting_frequency", "calendar_total_f82"),
            ("calendar_total", "calendar_N82"),
        ]
        for j, (base, alternative) in enumerate(comparisons):
            p0, p1 = (preds[base], preds[alternative])
            gain = dict(
                case=case,
                lead_h=lead,
                base=base,
                alternative=alternative,
                brier_gain=float(np.mean((y - p0) ** 2 - (y - p1) ** 2)),
                precision_gain=float(
                    y[rank_selection(p1)].mean() - y[rank_selection(p0)].mean()
                ),
                **interval(y, p0, p1, start[test], SEED + j),
            )
            gains.append(gain)
        prediction.to_csv(
            OUT / f"{case}_test_predictions.csv.gz", index=False, compression="gzip"
        )
        cases[case] = prediction
        print(
            case,
            "fit/cal/test",
            int(fit.sum()),
            int(cal.sum()),
            int(test.sum()),
            "high",
            int(y.sum()),
            flush=True,
        )
        return prediction

    features = original.block_features(blocks, 3)
    eligible3 = base_eligible & blocks.origin_valid_3h.to_numpy(bool)
    primary = run_case("primary_mean", y_mean, eligible3, features)
    frozen = pd.read_csv(OLD / "sgp_six_hour_state_test_predictions.csv.gz")
    frozen = frozen[
        (frozen.min_hours == 4) & (frozen.lead_h == 3) & (frozen["tail"] == "high")
    ]
    assert len(primary) == len(frozen) == 1850 and int(primary.event.sum()) == 336
    assert np.array_equal(
        pd.DatetimeIndex(primary.block_start),
        pd.DatetimeIndex(pd.to_datetime(frozen.block_start, utc=True)),
    )
    for name in features:
        assert np.allclose(
            primary["p_" + name], frozen["p_" + name], rtol=0, atol=1e-10
        ), name
    origin3 = pd.DatetimeIndex(blocks.origin_3h)
    current = hourly.reindex(origin3).to_numpy(float)
    history = np.column_stack(
        [
            hourly.reindex(origin3 - pd.Timedelta(hours=k)).to_numpy(float)
            for k in range(6)
        ]
    )
    nhist = np.isfinite(history).sum(axis=1)
    history_mean = np.divide(
        np.nansum(history, axis=1),
        nhist,
        out=np.full(len(blocks), np.nan),
        where=nhist > 0,
    )
    for label, value, valid in [
        ("current_ccn", current, np.isfinite(current)),
        ("recent6_ccn", history_mean, nhist >= 4),
    ]:
        extra = np.log1p(value)[:, None]
        f = {
            **features,
            "calendar_ccn": np.column_stack([features["calendar"], extra]),
            "calendar_total_ccn": np.column_stack([features["calendar_total"], extra]),
            "calendar_total_ccn_f82": np.column_stack(
                [features["calendar_total_f82"], extra]
            ),
        }
        run_case(
            label,
            y_mean,
            eligible3 & valid,
            f,
            pairs=[
                ("calendar_total", "calendar_total_f82"),
                ("calendar_ccn", "calendar_total_ccn_f82"),
                ("calendar_total_ccn", "calendar_total_ccn_f82"),
                ("calendar_total", "calendar_total_ccn"),
                ("calendar_total_f82", "calendar_total_ccn_f82"),
            ],
        )
    row = d["ts"].get_indexer(origin3)
    for cutoff in (80, 100):
        f = dict(features)
        fraction = np.full(len(blocks), np.nan)
        count = np.full(len(blocks), np.nan)
        valid = blocks.origin_valid_3h.to_numpy(bool)
        count[valid], fraction[valid] = original.spectra.fraction_above(
            d["bins"][row[valid]], d["diameter"], cutoff
        )
        f["calendar_total_f" + str(cutoff)] = np.column_stack(
            [features["calendar_total"], fraction]
        )
        f["calendar_N" + str(cutoff)] = np.column_stack(
            [features["calendar"], np.log1p(count)]
        )
        run_case(
            f"cutoff_{cutoff}",
            y_mean,
            eligible3,
            f,
            pairs=[
                ("calendar_total", "calendar_total_f" + str(cutoff)),
                ("calendar_total", "calendar_N" + str(cutoff)),
                ("calendar_N82", "calendar_N" + str(cutoff)),
            ],
        )
    run_case("median_target", y_median, eligible3, features)
    run_case(
        "sustained_target",
        y_sustained,
        eligible3 & (blocks.n_valid_ccn_hours.to_numpy() == 6),
        features,
    )
    for lead in (1, 3, 6):
        run_case(
            f"common_block_lead{lead}",
            y_mean,
            common,
            original.block_features(blocks, lead),
            lead,
            True,
        )
    for name, df in [
        ("scores", scores),
        ("paired_gains", gains),
        ("split_support", stages),
        ("reliability", calibration),
        ("monthly_coverage", coverage),
        ("boundary_checks", purge_rows),
    ]:
        pd.DataFrame(df).to_csv(OUT / f"{name}.csv", index=False)
    blocks.to_csv(
        OUT / "target_blocks_with_sensitivity_labels.csv.gz",
        index=False,
        compression="gzip",
    )
    monthly["median_target_q75"] = med_table
    monthly["sustained_hourly_q75"] = hourly_q75
    monthly.to_csv(OUT / "monthly_thresholds.csv")
    primary["selected_N"] = False
    primary["selected_size"] = False
    primary.loc[
        primary.index[rank_selection(primary.p_calendar_total.to_numpy())], "selected_N"
    ] = True
    primary.loc[
        primary.index[rank_selection(primary.p_calendar_total_f82.to_numpy())],
        "selected_size",
    ] = True
    primary.to_csv(OUT / "primary_selection_overlap.csv", index=False)
    overlap = []
    for a, b, label in [
        (True, True, "both"),
        (True, False, "N_only"),
        (False, True, "size_only"),
        (False, False, "neither"),
    ]:
        x = primary[(primary.selected_N == a) & (primary.selected_size == b)]
        overlap.append(
            dict(
                selection=label,
                n_blocks=len(x),
                high_blocks=int(x.event.sum()),
                median_ccn_cm3=float(x.mean_ccn_cm3.median()),
                median_hours_ge_threshold=float(x.hours_ge_mean_threshold.median()),
                median_max_hour_share=float(x.max_hour_share_of_sum.median()),
            )
        )
    pd.DataFrame(overlap).to_csv(OUT / "selection_overlap_summary.csv", index=False)
    manifest = dict(
        status="completed",
        primary_probabilities_reproduced=True,
        source_manifest_sha256=sha(OLD / "manifest.json"),
        analysis_code_sha256=sha(Path(__file__)),
        script_sha256=sha(Path(__file__)),
        cases=list(cases),
        preprocessing_gate_passed=True,
        source_pnsd_audit_sha256=d["mapped_audit_sha256"],
        split=dict(calibration=str(cal_cut), outer_test=str(outer)),
        review_scope="exploratory follow-up in opened archive; no test-selected settings",
        output_sha256={p.name: sha(p) for p in sorted(OUT.glob("*.csv*"))},
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
