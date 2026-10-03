"""Comment 17: fixed-prediction selection checks on archived CCN block outputs."""

from pathlib import Path
import os
from runtime import ROOT
from runtime import WORK

for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
    os.environ[k] = str(WORK)
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np, pandas as pd, json, hashlib, math
from sklearn.metrics import average_precision_score

OUT = ROOT / "results/proxy_and_size_range/month_relative"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
gains = []
provenance = []


def select(p, months, mode):
    selected = np.zeros(len(p), bool)
    groups = (
        [np.arange(len(p))]
        if mode == "global"
        else [np.flatnonzero(months == m) for m in np.unique(months)]
    )
    for ids in groups:
        selected[
            ids[np.argsort(-p[ids], kind="stable")[: math.ceil(0.1 * len(ids))]]
        ] = True
    return selected


def contrasts(y, r, pa, pb, months, mode):
    a, b = (select(pa, months, mode), select(pb, months, mode))
    only_a, only_b = (a & ~b, b & ~a)
    return [
        float(y[b].mean() - y[a].mean()),
        float(np.median(r[b]) - np.median(r[a])),
        (
            float(np.median(r[only_b]) - np.median(r[only_a]))
            if only_a.any() and only_b.any()
            else np.nan
        ),
    ]


for si, site in enumerate(("SGP", "ENA", "GUC", "MAO")):
    path = ROOT / f"results/cross_site/{site}/monthly_lead3_predictions.csv.gz"
    parent = json.loads((path.parent / "manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == parent["outputs"][path.name]
    d = pd.read_csv(path)
    ts = pd.to_datetime(d.block_start, utc=True)
    assert np.array_equal(np.argsort(ts.to_numpy()), np.arange(len(d))) and (
        not ts.duplicated().any()
    )
    y = d.event.to_numpy(int)
    c = d.mean_ccn_cm3.to_numpy(float)
    q = d.threshold_cm3.to_numpy(float)
    assert np.all(q > 0) and np.array_equal((c >= q).astype(int), y)
    ratio = c / q
    months = np.asarray(ts.dt.strftime("%Y-%m"))
    pa = d.p_calendar_total.to_numpy()
    pb = d.p_calendar_total_f82.to_numpy()
    provenance.append(
        {
            "site": site,
            "source": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_test": len(d),
            "average_precision_total": float(average_precision_score(y, pa)),
            "average_precision_size": float(average_precision_score(y, pb)),
        }
    )
    if site == "SGP":
        assert (
            int(y[select(pa, months, "global")].sum()) == 50
            and int(y[select(pb, months, "global")].sum()) == 108
        )
    weeks = ts.astype("int64").to_numpy() // (7 * 86400 * 10**9)
    groups = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    for mode in ("global", "within_year_month"):
        a, b = (select(pa, months, mode), select(pb, months, mode))
        for label, mask in [
            ("total_all", a),
            ("size_all", b),
            ("total_exclusive", a & ~b),
            ("size_exclusive", b & ~a),
        ]:
            rows.append(
                dict(
                    site=site,
                    ranking=mode,
                    selection=label,
                    n_selected=int(mask.sum()),
                    high_blocks=int(y[mask].sum()),
                    precision=float(y[mask].mean()),
                    median_ccn_cm3=float(np.median(c[mask])),
                    median_relative_ccn=float(np.median(ratio[mask])),
                    relative_q25=float(np.quantile(ratio[mask], 0.25)),
                    relative_q75=float(np.quantile(ratio[mask], 0.75)),
                )
            )
        original = contrasts(y, ratio, pa, pb, months, mode)
        rng = np.random.default_rng(20261002 + si * 100 + (mode != "global"))
        boot = []
        for _ in range(1000):
            ids = np.concatenate(
                [groups[j] for j in rng.integers(len(groups), size=len(groups))]
            )
            boot.append(
                contrasts(y[ids], ratio[ids], pa[ids], pb[ids], months[ids], mode)
            )
        boot = np.asarray(boot)
        for j, metric in enumerate(
            (
                "precision_gain",
                "median_relative_ccn_gain",
                "exclusive_median_relative_ccn_gain",
            )
        ):
            lo, hi = np.nanquantile(boot[:, j], [0.025, 0.975])
            gains.append(
                dict(
                    site=site,
                    ranking=mode,
                    metric=metric,
                    gain=original[j],
                    lo95=lo,
                    hi95=hi,
                    valid_draws=int(np.isfinite(boot[:, j]).sum()),
                    resamples=1000,
                    calendar_week_blocks=len(groups),
                )
            )
pd.DataFrame(rows).to_csv(OUT / "selection_summaries.csv", index=False)
pd.DataFrame(gains).to_csv(OUT / "paired_gains.csv", index=False)
manifest = {
    "sources": provenance,
    "definition": "measured block mean / archived fitting-calendar-month 75th percentile",
    "ranking": "top ceil(10% n) globally or separately within each observed UTC year-month",
    "bootstrap": "1000 paired seven-day calendar-block draws; fixed predictions and thresholds; rankings recalculated",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(gains).to_string(index=False))
print("AP", provenance[0])
