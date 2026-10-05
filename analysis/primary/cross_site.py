"""Cross-site six-hour high-CCN tests on frozen, audited aerosol records."""

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
import argparse, json, math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from analysis.diagnostics import temporal as audited
from analysis.primary import environmental as orig

for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[k] = str(WORK)
OUT = ROOT / "results/cross_site"
SITES = orig.builder.SITES


def rank(p):
    return np.argsort(-p, kind="stable")[: math.ceil(len(p) * 0.1)]


def scores(y, p):
    ix = rank(p)
    hits = int(y[ix].sum())
    return dict(
        n_test=len(y),
        high_blocks=int(y.sum()),
        brier=float(brier_score_loss(y, p)),
        average_precision=float(average_precision_score(y, p)) if y.sum() else None,
        selected=len(ix),
        hits=hits,
        precision=hits / len(ix),
        recall=hits / y.sum() if y.sum() else None,
    )


def intervals(y, a, b, dates, seed):
    weeks = pd.DatetimeIndex(dates).asi8 // (7 * 86400 * 10**9)
    groups = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    rng = np.random.default_rng(seed)
    bs = []
    pr = []
    for _ in range(1000):
        ix = np.concatenate(
            [groups[j] for j in rng.integers(len(groups), size=len(groups))]
        )
        z, p, q = (y[ix], a[ix], b[ix])
        bs.append(np.mean((z - p) ** 2 - (z - q) ** 2))
        pr.append(z[rank(q)].mean() - z[rank(p)].mean())
    return dict(
        brier_lo95=float(np.quantile(bs, 0.025)),
        brier_hi95=float(np.quantile(bs, 0.975)),
        precision_lo95=float(np.quantile(pr, 0.025)),
        precision_hi95=float(np.quantile(pr, 0.975)),
        n_weeks=len(groups),
        draws=1000,
    )


def blocks_for(d):
    c = orig.builder.load_ccn_instrument_only()[d["site"]]
    y = pd.to_numeric(c.N_CCN_mean_STP_B, errors="coerce")
    good = np.isfinite(y) & (y >= 0) & c.ccn_instrument_ok.astype(bool)
    hourly = pd.Series(
        y[good].to_numpy(float), index=pd.DatetimeIndex(c.loc[good, "timestamp"])
    )
    assert not hourly.index.has_duplicates
    starts = pd.date_range(
        hourly.index.min().floor("6h"), hourly.index.max().floor("6h"), freq="6h"
    )
    vals = np.column_stack(
        [
            hourly.reindex(starts + pd.Timedelta(hours=k)).to_numpy(float)
            for k in range(6)
        ]
    )
    n = np.isfinite(vals).sum(axis=1)
    means = np.divide(
        np.nansum(vals, axis=1), n, out=np.full(len(n), np.nan), where=n > 0
    )
    f = pd.DataFrame(
        dict(
            block_start=starts,
            block_end=starts + pd.Timedelta(hours=6),
            n_valid_ccn_hours=n,
            mean_ccn_cm3=means,
            month=starts.month,
            season=orig.season_of(starts.month),
        )
    )
    for h in (1, 3, 6):
        origins = starts - pd.Timedelta(hours=h)
        ix = d["ts"].get_indexer(origins)
        valid = ix >= 0
        valid[valid] &= d["current_ok"][ix[valid]]
        f[f"origin_{h}h"] = origins
        f[f"origin_valid_{h}h"] = valid
        for key, src in [("N15_300", "total"), ("f82", "f82")]:
            z = np.full(len(f), np.nan)
            z[valid] = d[src][ix[valid]]
            f[f"{key}_{h}h"] = z
    return f


def main(site):
    out = OUT / site
    out.mkdir(parents=True, exist_ok=True, mode=448)
    if (out / "manifest.json").exists():
        raise FileExistsError(out)
    d = audited.read_site(site)
    b = blocks_for(d)
    outer = d["cut"]
    common = (b.n_valid_ccn_hours >= 4) & (b.block_end <= outer)
    for h in (1, 3, 6):
        common &= b[f"origin_valid_{h}h"]
    dates = pd.DatetimeIndex(
        b.loc[common, "block_start"].dt.floor("D").unique()
    ).sort_values()
    manifest = dict(
        site=site,
        outer_boundary=str(outer),
        source_files=d["source_files"],
        mapped_audit_file=d["mapped_audit_file"],
        conservation=d["conservation"],
        preprocessing_gate_passed=True,
        pretest_common_days=len(dates),
        analysis_code_file=audited.file_info(Path(__file__)),
    )
    if len(dates) < 10:
        manifest.update(status="insufficient_pretest_dates")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return
    cut = dates[int(0.8 * len(dates))]
    threshold_fit = common & (b.block_end <= cut)
    th = (
        b.loc[threshold_fit]
        .groupby("month")
        .agg(
            n_fit_blocks=("mean_ccn_cm3", "size"),
            threshold_cm3=("mean_ccn_cm3", lambda x: x.quantile(0.75)),
        )
    )
    th["supported"] = th.n_fit_blocks >= 30
    th.to_csv(out / "monthly_threshold_support.csv")
    pooled = float(b.loc[threshold_fit, "mean_ccn_cm3"].quantile(0.75))
    manifest.update(
        calibration_boundary=str(cut),
        threshold_fit_blocks=int(threshold_fit.sum()),
        supported_months=[int(x) for x in th.index[th.supported]],
        pooled_threshold=pooled,
    )
    rows = []
    gains = []
    support = []
    unique = []
    seasonal = []
    for definition in ("monthly", "pooled_sensitivity"):
        threshold = (
            b.month.map(th.threshold_cm3.where(th.supported)).to_numpy(float)
            if definition == "monthly"
            else np.full(len(b), pooled)
        )
        y = (b.mean_ccn_cm3.to_numpy() >= threshold).astype(int)
        for h in (1, 3, 6):
            eligible = (
                (b.n_valid_ccn_hours >= 4)
                & b[f"origin_valid_{h}h"]
                & np.isfinite(threshold)
            )
            fit = eligible & (b.block_end <= cut)
            test = eligible & (b[f"origin_{h}h"] >= outer)
            pretest = (
                (b.n_valid_ccn_hours >= 4)
                & b[f"origin_valid_{h}h"]
                & (b[f"origin_{h}h"] >= outer)
            )
            status = (
                "scored"
                if fit.sum() >= 40
                and test.sum() >= 20
                and (len(np.unique(y[fit])) == 2)
                else "insufficient_support"
            )
            meta = dict(
                site=site,
                definition=definition,
                lead_h=h,
                n_fit=int(fit.sum()),
                n_test=int(test.sum()),
                high_test=int(y[test].sum()),
                test_blocks_before_threshold_support=int(pretest.sum()),
                excluded_unsupported_month=int(pretest.sum() - test.sum()),
                status=status,
            )
            support.append(meta)
            if status != "scored":
                continue
            assert (b.loc[fit, "block_end"] <= cut).all() and (
                b.loc[test, f"origin_{h}h"] >= outer
            ).all()
            xsets = orig.block_features(b, h)
            pred = b.loc[
                test,
                [
                    "block_start",
                    "block_end",
                    "month",
                    "season",
                    "mean_ccn_cm3",
                    "n_valid_ccn_hours",
                ],
            ].copy()
            pred["origin"] = b.loc[test, f"origin_{h}h"]
            pred["threshold_cm3"] = threshold[test]
            pred["event"] = y[test]
            yt = y[test]
            prior = float(y[fit].mean())
            pred["p_fitting_frequency"] = prior
            for name, x in xsets.items():
                pred["p_" + name] = (
                    orig.classifier().fit(x[fit], y[fit]).predict_proba(x[test])[:, 1]
                )
            prior_bs = brier_score_loss(yt, np.full(len(yt), prior))
            for col in [c for c in pred if c.startswith("p_")]:
                s = scores(yt, pred[col].to_numpy())
                s["brier_skill"] = 1 - s["brier"] / prior_bs if prior_bs else None
                if col == "p_fitting_frequency":
                    s.update(selected=None, hits=None, precision=None, recall=None)
                rows.append(
                    dict(
                        site=site,
                        definition=definition,
                        lead_h=h,
                        model=col[2:],
                        n_fit=int(fit.sum()),
                        **s,
                    )
                )
            for j, (a, c) in enumerate(
                [
                    ("calendar_total", "calendar_total_f82"),
                    ("calendar_total", "calendar_N82"),
                    ("fitting_frequency", "calendar_total_f82"),
                ]
            ):
                pa, pc = (pred["p_" + a].to_numpy(), pred["p_" + c].to_numpy())
                g = dict(
                    site=site,
                    definition=definition,
                    lead_h=h,
                    baseline=a,
                    added=c,
                    brier_gain=float(np.mean((yt - pa) ** 2 - (yt - pc) ** 2)),
                    precision_gain=float(yt[rank(pc)].mean() - yt[rank(pa)].mean()),
                )
                g.update(
                    intervals(
                        yt,
                        pa,
                        pc,
                        pred.block_start,
                        280900 + SITES.index(site) * 100 + h * 10 + j,
                    )
                )
                gains.append(g)
            sel0 = np.zeros(len(pred), bool)
            sel1 = sel0.copy()
            sel0[rank(pred.p_calendar_total.to_numpy())] = True
            sel1[rank(pred.p_calendar_total_f82.to_numpy())] = True
            for label, m in [
                ("both", sel0 & sel1),
                ("total_only", sel0 & ~sel1),
                ("size_only", sel1 & ~sel0),
                ("neither", ~sel0 & ~sel1),
            ]:
                unique.append(
                    dict(
                        site=site,
                        definition=definition,
                        lead_h=h,
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
            for season in orig.SEASONS:
                m = pred.season == season
                for model in ("calendar_total", "calendar_total_f82"):
                    if m.any():
                        seasonal.append(
                            dict(
                                site=site,
                                definition=definition,
                                lead_h=h,
                                season=season,
                                model=model,
                                **scores(yt[m], pred.loc[m, "p_" + model].to_numpy()),
                            )
                        )
            pred.to_csv(out / f"{definition}_lead{h}_predictions.csv.gz", index=False)
            if site == "SGP" and definition == "monthly" and (h == 3):
                assert len(pred) == 1850 and yt.sum() == 336
                assert [
                    scores(yt, pred["p_" + k].to_numpy())["hits"]
                    for k in ("calendar_total", "calendar_total_f82", "calendar_N82")
                ] == [50, 108, 109]
                old = pd.read_csv(
                    ROOT
                    / "results/block_robustness/primary_mean_test_predictions.csv.gz"
                )
                assert np.array_equal(
                    pd.to_datetime(old.block_start, utc=True),
                    pd.to_datetime(pred.block_start, utc=True),
                )
                for k in xsets:
                    assert np.allclose(
                        old["p_" + k], pred["p_" + k], rtol=0, atol=1e-10
                    )
                manifest["SGP_regression_test_passed"] = True
            print(
                site,
                definition,
                h,
                "fit/test",
                int(fit.sum()),
                len(pred),
                "high",
                int(yt.sum()),
                flush=True,
            )
    for name, records in [
        ("scores", rows),
        ("paired_gains", gains),
        ("eligibility", support),
        ("selected_set_concentrations", unique),
        ("seasonal_scores", seasonal),
    ]:
        pd.DataFrame(records).to_csv(out / (name + ".csv"), index=False)
    b.to_csv(out / "blocks.csv.gz", index=False)
    manifest.update(
        status="complete",
        script_file=audited.file_info(Path(__file__)),
        outputs={p.name: audited.file_info(p) for p in out.iterdir() if p.is_file()},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True, choices=SITES)
    main(p.parse_args().site)
