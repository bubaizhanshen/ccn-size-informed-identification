"""Current-manuscript proxy-persistence and upper-range comparisons; no document edits."""

from pathlib import Path
import os
from runtime import ROOT
from runtime import WORK

os.umask(63)
WORK.mkdir(parents=True, exist_ok=True, mode=448)


def private_env():
    for k in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
        "MPLCONFIGDIR",
    ):
        os.environ[k] = str(WORK)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


private_env()
import argparse, json, math, hashlib, sys, time
import numpy as np
import pandas as pd
import sklearn, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, brier_score_loss
import shared_temporal as a
import diagnostic_size_fraction as old
import primary_cross_site as blocks
import shared_pnsd_preflight as pre

private_env()
OUT = ROOT / "results/proxy_and_size_range"
SITES = old.SITES
LGB_LEADS = (1, 3, 6, 12, 24)
PARAM = dict(
    n_estimators=160,
    num_leaves=15,
    learning_rate=0.03,
    min_child_samples=40,
    reg_lambda=1.0,
    random_state=20261002,
    n_jobs=1,
    verbosity=-1,
    deterministic=True,
    force_col_wise=True,
)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, float) and (not np.isfinite(x)):
            return None
        return x

    path.write_text(
        json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False)
    )


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def regression(family):
    if family == "ridge":
        return old.regressor()
    return lgb.LGBMRegressor(**PARAM)


def classification(family):
    if family == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1, max_iter=2000)),
            ]
        )
    return lgb.LGBMClassifier(**PARAM)


def fitpred(x, y, fit, score, family):
    assert fit.sum() >= 40 and score.sum() > 0 and np.isfinite(y[fit]).all()
    keep = np.isfinite(x[fit]).any(axis=0)
    model = regression(family).fit(x[fit][:, keep], y[fit])
    return model.predict(x[score][:, keep])


def oof_proxy(x, y, valid, ts, end, family):
    pool = valid & np.asarray(ts < end)
    days = pd.DatetimeIndex(ts[pool].floor("D").unique()).sort_values()
    assert len(days) >= 10
    boundaries = [
        days[min(len(days) - 1, int(len(days) * q))] for q in (0.2, 0.4, 0.6, 0.8)
    ] + [end]
    result = np.full(len(ts), np.nan)
    records = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        fit = pool & np.asarray(ts < left)
        score = pool & np.asarray((ts >= left) & (ts < right))
        assert fit.sum() >= 40 and score.sum() > 0
        result[score] = fitpred(x, y, fit, score, family)
        records.append(
            dict(
                start=str(left),
                end=str(right),
                n_fit=int(fit.sum()),
                n_oof=int(score.sum()),
                max_fit_label=str(ts[fit].max()),
                min_pred_origin=str(ts[score].min()),
            )
        )
    return (result, records)


def error_interval(y, p0, p1, ts, seed, draws=1000):
    group = ts.asi8 // (7 * 86400 * 10**9)
    _, ix = np.unique(group, return_inverse=True)
    n = np.bincount(ix)
    sy = np.bincount(ix, weights=y)
    sy2 = np.bincount(ix, weights=y * y)
    e0 = np.bincount(ix, weights=(y - p0) ** 2)
    e1 = np.bincount(ix, weights=(y - p1) ** 2)
    raw = np.expm1(y)
    ed = np.bincount(
        ix, weights=np.abs(raw - np.expm1(p0)) - np.abs(raw - np.expm1(p1))
    )
    rng = np.random.default_rng(seed)
    take = rng.integers(len(n), size=(draws, len(n)))
    count = n[take].sum(1)
    sst = sy2[take].sum(1) - sy[take].sum(1) ** 2 / count
    rr = (e0[take].sum(1) - e1[take].sum(1)) / sst
    mm = ed[take].sum(1) / count
    return dict(
        delta_r2=r2(y, p1) - r2(y, p0),
        lo95=float(np.quantile(rr, 0.025)),
        hi95=float(np.quantile(rr, 0.975)),
        mae_gain_cm3=float(
            np.mean(np.abs(raw - np.expm1(p0)) - np.abs(raw - np.expm1(p1)))
        ),
        mae_lo95=float(np.quantile(mm, 0.025)),
        mae_hi95=float(np.quantile(mm, 0.975)),
        n_week_groups=len(n),
        draws=draws,
    )


def score_reg(y, p):
    return dict(
        r2_log=r2(y, p),
        mae_log=float(np.abs(y - p).mean()),
        mae_cm3=float(np.abs(np.expm1(y) - np.expm1(p)).mean()),
        mean_bias_cm3=float((np.expm1(p) - np.expm1(y)).mean()),
    )


def prepare(site, d, dest):
    paths = a.builder.source_smps(site)
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    names = [n for n in frames[0] if n.startswith("D_")]
    diam = np.array([float(n[2:].replace("_", ".")) for n in names])
    order = np.argsort(diam)
    diam = diam[order]
    names = [names[k] for k in order]
    f = pd.concat(
        [p[names + ["N_CN_SMPS_STP", "Start_date"]] for p in frames], ignore_index=True
    )
    del frames
    density = f[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    native_edges = pre.log10_bin_edges(diam)
    needed = (native_edges[1:] > np.log10(15)) & (native_edges[:-1] < np.log10(500))
    native_valid = (
        np.isfinite(density[:, needed]).all(1)
        & (density[:, needed] >= 0).all(1)
        & ~(density < 0).any(1)
    )
    supplied = pd.to_numeric(f.N_CN_SMPS_STP, errors="coerce").to_numpy(float)
    edges = np.linspace(np.log10(15.0), np.log10(500.0), 129)
    grid = 10 ** ((edges[1:] + edges[:-1]) / 2)
    report, _, mapped_valid = pre.audit(
        density, diam, native_valid, "dndlog10dp", grid, supplied_total=supplied
    )
    contract = dict(
        site=site,
        source_sha256=d["source_hashes"],
        source_file="Figshare harmonized SMPS D_* columns",
        input_semantics="dndlog10dp",
        concentration_unit="particles cm-3 per log10 diameter",
        diameter_unit="nm",
        diameter_definition="midpoint",
        native_lower_nm=float(diam.min()),
        native_upper_nm=float(diam.max()),
        native_channel_count=len(diam),
        density_log_base=10,
        native_width_method="log-midpoint-inferred integration cells for density",
        semantics_evidence="Current manuscript source/archive documentation and original nine-site independent-total audit",
        timezone="UTC",
        instrument_metadata_source="Current manuscript Table S1 and Scientific Data 2025 archive metadata",
        scan_duration="site-specific original Table S1",
        negative_value_policy="invalidate rows with any negative native density; no clipping",
        missing_value_policy="complete finite native density cells overlapping 15–500 nm; absent cells outside this comparison are not imputed",
        target_scope="15–500 nm; original 24-bin 15–300 nm predictors remain unchanged",
    )
    report.update(
        metadata_gate_passed=True,
        representation_contract=contract,
        input_sources_sha256=d["source_hashes"],
        snapshot_sha256=sha(ROOT / str(Path(pre.__file__).resolve())),
    )
    pd.DataFrame(
        dict(
            diameter_nm=diam,
            n_finite=np.isfinite(density).sum(0),
            n_nonnegative=((density >= 0) & np.isfinite(density)).sum(0),
        )
    ).to_csv(dest / "native_channel_coverage.csv", index=False)
    dump(dest / "upper500_preflight.json", report)
    if int(native_valid.sum()) == 0:
        dump(
            dest / "upper500_unavailable.json",
            dict(
                site=site,
                status="no_complete_15_500_native_rows",
                reason="Listed archive grid is not complete measured coverage at this station; do not impute missing high-diameter cells",
                native_channel_coverage="native_channel_coverage.csv",
            ),
        )
        return dict(
            valid=np.zeros(len(d["ts"]), bool),
            total=np.full(len(d["ts"]), np.nan),
            n82=np.full(len(d["ts"]), np.nan),
            f82=np.full(len(d["ts"]), np.nan),
            tail=np.full(len(d["ts"]), np.nan),
        )
    assert report["preprocessing_gate_passed"], (
        site,
        "500 preflight failed",
        report["conservation_mapped_over_native"],
    )
    tail = pre.integrate_native_range(
        density, diam, "dndlog10dp", np.log10(300), np.log10(500)
    )
    native500 = pre.integrate_native_range(
        density, diam, "dndlog10dp", np.log10(15), np.log10(500)
    )
    source_ts = pd.DatetimeIndex(pd.to_datetime(f.Start_date, utc=True))
    assert len(source_ts) == len(tail)
    ix = d["raw_index"].reindex(d["ts"]).to_numpy(int)
    assert np.array_equal(source_ts[ix], d["ts"])
    valid = d["current_ok"] & native_valid[ix] & mapped_valid[ix]
    tail_aligned = tail[ix]
    tail_aligned[~valid] = np.nan
    n500 = d["total"] + tail_aligned
    n82500 = d["n82"] + tail_aligned
    f500 = n82500 / n500
    ratio = n500[valid] / native500[ix][valid]
    q = np.quantile(ratio, [0.05, 0.5, 0.95])
    assert 0.95 <= q[0] <= 1.05 and 0.97 <= q[1] <= 1.03 and (0.95 <= q[2] <= 1.05), (
        site,
        q,
    )
    dump(
        dest / "upper500_representation.json",
        dict(
            **contract,
            n_original_valid=int(d["current_ok"].sum()),
            n_500_valid=int(valid.sum()),
            n500_over_native500_q05_median_q95=q.tolist(),
            operation="retain original N15–300 and N82–300; add native log-cell-overlap integral of 300–500 density",
            source_raw_rows=len(f),
            preflight_sha256=sha(dest / "upper500_preflight.json"),
        ),
    )
    tailrows = []
    for period, m in [
        ("all", valid),
        ("fit", valid & np.asarray(d["ts"] < d["cut"])),
        ("test", valid & np.asarray(d["ts"] >= d["cut"])),
    ] + [
        (s, valid & (d["ts"].month % 12 // 3 == j))
        for j, s in enumerate(("DJF", "MAM", "JJA", "SON"))
    ]:
        for name, value in [
            ("tail_over_N300", tail_aligned / d["total"]),
            ("tail_over_N82_300", tail_aligned / d["n82"]),
            ("tail_share_N500", tail_aligned / n500),
            ("fraction_change", f500 - d["f82"]),
        ]:
            z = value[m]
            z = z[np.isfinite(z)]
            tailrows.append(
                dict(
                    site=site,
                    period=period,
                    metric=name,
                    n=len(z),
                    q05=float(np.quantile(z, 0.05)) if len(z) else np.nan,
                    median=float(np.median(z)) if len(z) else np.nan,
                    q95=float(np.quantile(z, 0.95)) if len(z) else np.nan,
                )
            )
    pd.DataFrame(tailrows).to_csv(dest / "tail_contribution.csv", index=False)
    return dict(valid=valid, tail=tail_aligned, total=n500, n82=n82500, f82=f500)


def features(d, ext=None):
    aux = d["aux"][:, :-1]
    n = d["total"] if ext is None else ext["total"]
    f = d["f82"] if ext is None else ext["f82"]
    n82 = d["n82"] if ext is None else ext["n82"]
    return dict(
        total=np.column_stack((aux, np.log1p(n))),
        total_plus_f82=np.column_stack((aux, np.log1p(n), f)),
        N82=np.column_stack((aux, np.log1p(n82))),
    )


def hourly(site, d, ext, dest):
    ts, cut = (d["ts"], d["cut"])
    xsets = features(d)
    x500 = features(d, ext)
    aux = d["aux"][:, :-1]
    rows = []
    gains = []
    upper = []
    uppergains = []
    checks = []
    stages = []
    upper_support = []
    for family in ("ridge", "lightgbm"):
        leads = range(1, 25) if family == "ridge" else LGB_LEADS
        nowfit = d["eligible"][0] & np.asarray(ts < cut)
        for rep, x in xsets.items():
            score = np.asarray(ts >= cut) & d["current_ok"]
            now = np.full(len(ts), np.nan)
            now[score] = fitpred(x, d["y"][0], nowfit, score, family)
            oof, record = oof_proxy(x, d["y"][0], d["eligible"][0], ts, cut, family)
            stages.append(
                dict(
                    site=site,
                    family=family,
                    representation=rep,
                    n_current_fit=int(nowfit.sum()),
                    last_current_fit_label=str(ts[nowfit].max()),
                    outer_cut=str(cut),
                    folds=record,
                )
            )
            for h in leads:
                y = d["y"][h]
                fit = d["eligible"][h] & np.asarray(ts + pd.Timedelta(hours=h) < cut)
                test = d["eligible"][h] & np.asarray(ts >= cut)
                matched = fit & np.isfinite(oof)
                train_x = np.column_stack((oof, aux))
                pred_x = np.column_stack((now, aux))
                joined = train_x.copy()
                joined[test] = pred_x[test]
                predictions = dict(
                    proxy_persistence=now[test],
                    direct_full_fit=fitpred(x, y, fit, test, family),
                    direct_transition_matched=fitpred(x, y, matched, test, family),
                    proxy_transition=fitpred(joined, y, matched, test, family),
                )
                meta = dict(
                    site=site,
                    family=family,
                    representation=rep,
                    lead_h=h,
                    n_test=int(test.sum()),
                    n_direct_fit=int(fit.sum()),
                    n_matched_fit=int(matched.sum()),
                )
                for route, p in predictions.items():
                    rows.append(dict(**meta, route=route, **score_reg(y[test], p)))
                for j, (base, alt) in enumerate(
                    [
                        ("proxy_persistence", "direct_full_fit"),
                        ("proxy_transition", "direct_full_fit"),
                        ("proxy_transition", "direct_transition_matched"),
                    ]
                ):
                    gains.append(
                        dict(
                            **meta,
                            baseline=base,
                            alternative=alt,
                            **error_interval(
                                y[test],
                                predictions[base],
                                predictions[alt],
                                ts[test],
                                20261002 + h * 20 + j,
                            ),
                        )
                    )
                frame = pd.DataFrame(
                    dict(
                        timestamp=ts[test].astype(str),
                        target_log=y[test],
                        **{"pred_" + k: v for k, v in predictions.items()},
                    )
                )
                frame.to_csv(dest / f"hourly_{family}_{rep}_{h}h.csv.gz", index=False)
                if family == "ridge":
                    archive = (
                        ROOT
                        / f"results/ccn_v24_controls_20260928/{site}/minimal_anchor_{h}h_predictions.csv.gz"
                    )
                    oldp = pd.read_csv(archive)
                    assert np.array_equal(
                        pd.to_datetime(oldp.timestamp, utc=True), ts[test]
                    )
                    delta = float(
                        np.max(
                            np.abs(
                                oldp["pred_" + rep].to_numpy()
                                - predictions["direct_full_fit"]
                            )
                        )
                    )
                    assert delta < 1e-09, (
                        site,
                        rep,
                        h,
                        "old direct not reproduced",
                        delta,
                    )
                    checks.append(
                        dict(
                            site=site,
                            representation=rep,
                            lead_h=h,
                            max_abs_prediction_error=delta,
                            source_sha256=sha(archive),
                        )
                    )
        if not ext["valid"].any():
            continue
        for h in leads:
            y = d["y"][h]
            valid = d["eligible"][h] & ext["valid"]
            fit = valid & np.asarray(ts + pd.Timedelta(hours=h) < cut)
            test = valid & np.asarray(ts >= cut)
            supported = bool(fit.sum() >= 40 and test.sum() > 0)
            upper_support.append(
                dict(
                    site=site,
                    family=family,
                    lead_h=h,
                    n_fit=int(fit.sum()),
                    n_test=int(test.sum()),
                    supported=supported,
                )
            )
            if not supported:
                continue
            predictions = {}
            for uppernm, sets in [(300, xsets), (500, x500)]:
                for rep, x in sets.items():
                    p = fitpred(x, y, fit, test, family)
                    predictions[f"{rep}_{uppernm}"] = p
                    upper.append(
                        dict(
                            site=site,
                            family=family,
                            upper_nm=uppernm,
                            representation=rep,
                            lead_h=h,
                            n_fit=int(fit.sum()),
                            n_test=int(test.sum()),
                            **score_reg(y[test], p),
                        )
                    )
                for rep in ("total_plus_f82", "N82"):
                    uppergains.append(
                        dict(
                            site=site,
                            family=family,
                            lead_h=h,
                            comparison="size_vs_total",
                            representation=rep,
                            upper_nm=uppernm,
                            **error_interval(
                                y[test],
                                predictions[f"total_{uppernm}"],
                                predictions[f"{rep}_{uppernm}"],
                                ts[test],
                                20261002 + h * 30 + uppernm,
                            ),
                        )
                    )
            for rep in xsets:
                uppergains.append(
                    dict(
                        site=site,
                        family=family,
                        lead_h=h,
                        comparison="500_vs_300",
                        representation=rep,
                        upper_nm=500,
                        **error_interval(
                            y[test],
                            predictions[f"{rep}_300"],
                            predictions[f"{rep}_500"],
                            ts[test],
                            20261002 + h * 40 + len(rep),
                        ),
                    )
                )
            pd.DataFrame(
                dict(
                    timestamp=ts[test].astype(str),
                    target_log=y[test],
                    **{"pred_" + k: v for k, v in predictions.items()},
                )
            ).to_csv(dest / f"upper_{family}_{h}h.csv.gz", index=False)
            print(site, family, h, "hourly done", flush=True)
    for name, data in [
        ("hourly_scores", rows),
        ("hourly_route_gains", gains),
        ("upper_hourly_scores", upper),
        ("upper_hourly_gains", uppergains),
        ("old_prediction_reproduction", checks),
        ("upper_target_support", upper_support),
    ]:
        pd.DataFrame(data).to_csv(dest / (name + ".csv"), index=False)
    dump(dest / "proxy_chronology.json", stages)


def classify_predict(x, y, fit, test, family):
    assert len(np.unique(y[fit])) == 2 and np.isfinite(x[test]).any(axis=0).any()
    keep = np.isfinite(x[fit]).any(axis=0)
    return (
        classification(family)
        .fit(x[fit][:, keep], y[fit])
        .predict_proba(x[test][:, keep])[:, 1]
    )


def rank(p):
    return np.argsort(-p, kind="stable")[: math.ceil(0.1 * len(p))]


def score_block(y, p, c, relative):
    ix = rank(p)
    return dict(
        n_test=len(y),
        selected=len(ix),
        high_blocks=int(y.sum()),
        hits=int(y[ix].sum()),
        precision=float(y[ix].mean()),
        recall=float(y[ix].sum() / y.sum()),
        average_precision=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        median_ccn_cm3=float(np.median(c[ix])),
        median_relative_ccn=float(np.median(relative[ix])),
    )


def high_blocks(site, d, ext, dest):
    original = ROOT / f"results/cross_site/{site}"
    if not (original / "blocks.csv.gz").exists():
        return
    om = json.loads((original / "manifest.json").read_text())
    supported = om.get("supported_months", [])
    if om.get("status") != "complete" or not supported:
        return
    b = pd.read_csv(original / "blocks.csv.gz")
    th = pd.read_csv(original / "monthly_threshold_support.csv").set_index("month")
    for c in ("block_start", "block_end", "origin_1h", "origin_3h", "origin_6h"):
        b[c] = pd.to_datetime(b[c], utc=True)
    cut = pd.Timestamp(om["calibration_boundary"])
    outer = pd.Timestamp(om["outer_boundary"])
    thresholds = b.month.map(th.threshold_cm3.where(th.supported)).to_numpy()
    y = (b.mean_ccn_cm3.to_numpy() >= thresholds).astype(int)
    scores = []
    gains = []
    settings = []
    tailcoverage = []
    for family in ("logistic", "lightgbm"):
        regfamily = "ridge" if family == "logistic" else "lightgbm"
        for h in (1, 3, 6):
            oldsets = blocks.orig.block_features(b, h)
            cal = oldsets["calendar"]
            origin = pd.DatetimeIndex(b[f"origin_{h}h"])
            ix = d["ts"].get_indexer(origin)
            safe = np.maximum(ix, 0)
            good = b[f"origin_valid_{h}h"].to_numpy(bool)
            assert (ix[good] >= 0).all()
            for key, rep in [
                ("total", "calendar_total"),
                ("f82", "calendar_total_f82"),
            ]:
                col = -1
                assert np.allclose(
                    d[key][safe[good]],
                    (
                        oldsets[rep][good, col]
                        if key == "f82"
                        else np.expm1(oldsets[rep][good, col])
                    ),
                )
            eligible = (
                (b.n_valid_ccn_hours >= 4).to_numpy() & good & np.isfinite(thresholds)
            )
            fit = eligible & (b.block_end <= cut).to_numpy()
            test = eligible & (origin >= outer)
            hourts = d["ts"]
            calendar = np.column_stack(
                [
                    np.sin(2 * np.pi * hourts.hour / 24),
                    np.cos(2 * np.pi * hourts.hour / 24),
                    np.sin(2 * np.pi * hourts.dayofyear / 365.25),
                    np.cos(2 * np.pi * hourts.dayofyear / 365.25),
                ]
            )
            hourlysets = dict(
                total=np.column_stack((calendar, np.log1p(d["total"]))),
                total_plus_f82=np.column_stack(
                    (calendar, np.log1p(d["total"]), d["f82"])
                ),
                N82=np.column_stack((calendar, np.log1p(d["n82"]))),
            )
            modelnames = {
                "total": "calendar_total",
                "total_plus_f82": "calendar_total_f82",
                "N82": "calendar_N82",
            }
            frame = b.loc[
                test, ["block_start", "mean_ccn_cm3", "month", "season"]
            ].copy()
            frame["threshold_cm3"] = thresholds[test]
            frame["event"] = y[test]
            for rep, x in hourlysets.items():
                proxyfit = d["eligible"][0] & np.asarray(hourts < cut)
                allscore = d["current_ok"] & np.asarray(hourts >= cut)
                proxy = np.full(len(hourts), np.nan)
                proxy[allscore] = fitpred(x, d["y"][0], proxyfit, allscore, regfamily)
                oof, folds = oof_proxy(
                    x, d["y"][0], d["eligible"][0], hourts, cut, regfamily
                )
                value = oof[safe].copy()
                value[test] = proxy[safe[test]]
                matched = fit & (ix >= 0) & np.isfinite(value)
                px = np.column_stack((cal, value))
                direct = oldsets[modelnames[rep]]
                predictions = dict(
                    direct_full_fit=classify_predict(direct, y, fit, test, family),
                    direct_transition_matched=classify_predict(
                        direct, y, matched, test, family
                    ),
                    proxy_transition=classify_predict(px, y, matched, test, family),
                    proxy_persistence_ranking=proxy[safe[test]],
                )
                meta = dict(
                    site=site,
                    family=family,
                    representation=rep,
                    lead_h=h,
                    n_direct_fit=int(fit.sum()),
                    n_matched_fit=int(matched.sum()),
                )
                for route, p in predictions.items():
                    if route == "proxy_persistence_ranking":
                        take = rank(p)
                        result = dict(
                            n_test=int(test.sum()),
                            selected=len(take),
                            high_blocks=int(y[test].sum()),
                            hits=int(y[test][take].sum()),
                            precision=float(y[test][take].mean()),
                            recall=float(y[test][take].sum() / y[test].sum()),
                            average_precision=float(
                                average_precision_score(y[test], p)
                            ),
                            brier=None,
                            median_ccn_cm3=float(
                                frame.mean_ccn_cm3.iloc[take].median()
                            ),
                            median_relative_ccn=float(
                                np.median(
                                    frame.mean_ccn_cm3.to_numpy()[take]
                                    / thresholds[test][take]
                                )
                            ),
                        )
                    else:
                        result = score_block(
                            y[test],
                            p,
                            frame.mean_ccn_cm3.to_numpy(),
                            frame.mean_ccn_cm3.to_numpy() / thresholds[test],
                        )
                    scores.append(
                        dict(
                            **meta,
                            experiment="proxy",
                            upper_nm=300,
                            route=route,
                            **result,
                        )
                    )
                    frame[f"{rep}_{route}"] = p
                for j, route in enumerate(
                    ("direct_full_fit", "direct_transition_matched")
                ):
                    p0, p1 = (predictions["proxy_transition"], predictions[route])
                    gains.append(
                        dict(
                            **meta,
                            experiment="proxy",
                            baseline="proxy_transition",
                            alternative=route,
                            upper_nm=300,
                            brier_gain=float(
                                np.mean((y[test] - p0) ** 2 - (y[test] - p1) ** 2)
                            ),
                            precision_gain=float(
                                y[test][rank(p1)].mean() - y[test][rank(p0)].mean()
                            ),
                            **blocks.intervals(
                                y[test],
                                p0,
                                p1,
                                frame.block_start,
                                20261002 + h * 10 + j,
                            ),
                        )
                    )
                if family == "logistic":
                    previous = pd.read_csv(
                        original / f"monthly_lead{h}_predictions.csv.gz"
                    )
                    assert np.array_equal(
                        pd.to_datetime(previous.block_start, utc=True),
                        pd.DatetimeIndex(frame.block_start),
                    )
                    err = float(
                        np.max(
                            np.abs(
                                previous["p_" + modelnames[rep]].to_numpy()
                                - predictions["direct_full_fit"]
                            )
                        )
                    )
                    assert err < 1e-09, (site, h, rep, err)
                settings.append(
                    dict(
                        **meta,
                        proxy_fit_end=str(hourts[proxyfit].max()),
                        classification_fit_end=str(b.loc[fit, "block_end"].max()),
                        calibration_boundary=str(cut),
                        outer_boundary=str(outer),
                        proxy_folds=folds,
                    )
                )
            frame.to_csv(dest / f"block_proxy_{family}_{h}h.csv.gz", index=False)
            good500 = good & (ix >= 0) & ext["valid"][safe]
            fit500 = fit & good500
            test500 = test & good500
            tailcoverage.append(
                dict(
                    site=site,
                    lead_h=h,
                    n_fit_300=int(fit.sum()),
                    n_fit_common=int(fit500.sum()),
                    n_test_300=int(test.sum()),
                    n_test_common=int(test500.sum()),
                )
            )
            if fit500.sum() < 40 or test500.sum() == 0:
                continue
            newsets = dict(
                total=np.column_stack((cal, np.log1p(ext["total"][safe]))),
                total_plus_f82=np.column_stack(
                    (cal, np.log1p(ext["total"][safe]), ext["f82"][safe])
                ),
                N82=np.column_stack((cal, np.log1p(ext["n82"][safe]))),
            )
            uf = b.loc[
                test500, ["block_start", "mean_ccn_cm3", "month", "season"]
            ].copy()
            uf["event"] = y[test500]
            uf["threshold_cm3"] = thresholds[test500]
            preds = {}
            for up, sets in [
                (300, {k: oldsets[v] for k, v in modelnames.items()}),
                (500, newsets),
            ]:
                for rep, x in sets.items():
                    p = classify_predict(x, y, fit500, test500, family)
                    preds[f"{rep}_{up}"] = p
                    uf[f"p_{rep}_{up}"] = p
                    scores.append(
                        dict(
                            site=site,
                            family=family,
                            representation=rep,
                            lead_h=h,
                            n_direct_fit=int(fit500.sum()),
                            n_matched_fit=int(fit500.sum()),
                            experiment="upper",
                            upper_nm=up,
                            route="direct_common",
                            **score_block(
                                y[test500],
                                p,
                                uf.mean_ccn_cm3.to_numpy(),
                                uf.mean_ccn_cm3.to_numpy() / thresholds[test500],
                            ),
                        )
                    )
                for j, rep in enumerate(("total_plus_f82", "N82")):
                    p0, p1 = (preds[f"total_{up}"], preds[f"{rep}_{up}"])
                    gains.append(
                        dict(
                            site=site,
                            family=family,
                            representation=rep,
                            lead_h=h,
                            experiment="upper",
                            baseline="total",
                            alternative=rep,
                            upper_nm=up,
                            brier_gain=float(
                                np.mean((y[test500] - p0) ** 2 - (y[test500] - p1) ** 2)
                            ),
                            precision_gain=float(
                                y[test500][rank(p1)].mean()
                                - y[test500][rank(p0)].mean()
                            ),
                            **blocks.intervals(
                                y[test500],
                                p0,
                                p1,
                                uf.block_start,
                                20261002 + h * 10 + up + j,
                            ),
                        )
                    )
            uf.to_csv(dest / f"block_upper_{family}_{h}h.csv.gz", index=False)
            print(site, family, h, "blocks done", flush=True)
    for name, data in [
        ("block_scores", scores),
        ("block_gains", gains),
        ("block_upper_coverage", tailcoverage),
    ]:
        pd.DataFrame(data).to_csv(dest / (name + ".csv"), index=False)
    dump(dest / "block_chronology.json", settings)


def main(site):
    dest = OUT / site
    dest.mkdir(parents=True, exist_ok=True, mode=448)
    if (dest / "manifest.json").exists():
        raise FileExistsError(dest)
    start = time.time()
    d = a.read_site(site, leads=tuple(range(1, 25)))
    private_env()
    print(site, "existing cohort reproduced", flush=True)
    ext = prepare(site, d, dest)
    print(
        site,
        "500 nm status:",
        "passed" if ext["valid"].any() else "unavailable; proxy experiment proceeds",
        flush=True,
    )
    hourly(site, d, ext, dest)
    if site in ("SGP", "ENA", "GUC", "MAO"):
        high_blocks(site, d, ext, dest)
    manifest = dict(
        status="complete",
        site=site,
        completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        elapsed_seconds=time.time() - start,
        job_id=os.environ.get("SLURM_JOB_ID"),
        array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
        source_sha256=d["source_hashes"],
        mapped_audit_sha256=d["mapped_audit_sha256"],
        test_boundary=str(d["cut"]),
        script_sha256=sha(Path(__file__)),
        analysis_code_sha256=sha(Path(__file__)),
        software=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            pandas=pd.__version__,
            sklearn=sklearn.__version__,
            lightgbm=lgb.__version__,
        ),
        model_parameters=PARAM,
        outputs={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
    )
    dump(dest / "manifest.json", manifest)
    print(site, "COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--site", choices=SITES)
    p.add_argument("--index", type=int)
    p.add_argument("--attempt", type=int, default=1)
    args = p.parse_args()
    if args.attempt > 1:
        OUT = OUT / f"attempt{args.attempt}"
    main(args.site or SITES[args.index])
