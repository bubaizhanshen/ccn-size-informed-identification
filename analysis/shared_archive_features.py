"""Harmonized SMPS columns and size-band integration."""

from pathlib import Path
import os, re
import numpy as np
import pandas as pd
from runtime import ROOT

DATA = ROOT / "inputs/figshare_27913806/selected"


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_diameter(col: str) -> float:
    return float(col[2:].replace("_", "."))


def pnsd_band_values(df: pd.DataFrame) -> pd.DataFrame:
    """Integrate dN/dlog10(Dp) on exact log-overlap intervals.

    The harmonized files document the D-columns as dN/dlogD values.  Bin
    edges are reconstructed as geometric midpoints, with extrapolated outer
    edges.  Values are not interpolated across missing bins.
    """
    dcols = [c for c in df.columns if c.startswith("D_")]
    centers = np.asarray([parse_diameter(c) for c in dcols], float)
    logc = np.log10(centers)
    edges = np.empty(len(centers) + 1, float)
    edges[1:-1] = (logc[:-1] + logc[1:]) / 2
    edges[0] = logc[0] - (logc[1] - logc[0]) / 2
    edges[-1] = logc[-1] + (logc[-1] - logc[-2]) / 2
    values = df[dcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values[values < 0] = np.nan
    bands = {
        "n_15_25": (15.0, 25.0),
        "n_25_50": (25.0, 50.0),
        "n_50_100": (50.0, 100.0),
        "n_100_300": (100.0, 300.0),
        "n_300_500": (300.0, 500.0),
        "n_15_300": (15.0, 300.0),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        llo, lhi = (np.log10(lo), np.log10(hi))
        overlap = np.maximum(
            0.0, np.minimum(edges[1:], lhi) - np.maximum(edges[:-1], llo)
        )
        expected = overlap.sum()
        finite_width = np.where(np.isfinite(values), overlap[None, :], 0.0).sum(axis=1)
        result = np.nansum(values * overlap[None, :], axis=1)
        result[finite_width < 0.75 * expected] = np.nan
        out[name] = result
    out = pd.DataFrame(out, index=df.index)
    total = out["n_15_300"]
    for c in ["n_15_25", "n_25_50", "n_50_100", "n_100_300", "n_300_500"]:
        out[f"frac_{c[2:]}"] = out[c] / total.replace(0, np.nan)
    weights = values * np.diff(edges)[None, :]
    good = np.isfinite(weights) & (weights >= 0)
    den = np.where(good, weights, 0.0).sum(axis=1)
    numerator = np.where(good, weights * logc[None, :], 0.0).sum(axis=1)
    gmd_log = np.full_like(den, np.nan, dtype=float)
    np.divide(numerator, den, out=gmd_log, where=den > 0)
    out["pnsd_gmd"] = 10.0**gmd_log
    return out


def load_smps() -> dict[str, pd.DataFrame]:
    files = {}
    for p in sorted((DATA / "smps").glob("*_smps_hour*.csv")):
        m = re.match("([A-Za-z][A-Za-z0-9_]*?)_smps_hour", p.name)
        if m:
            files.setdefault(m.group(1).upper(), []).append(p)
    result = {}
    for site, paths in files.items():
        frames = []
        for p in paths:
            x = pd.read_csv(p, low_memory=False)
            x["timestamp"] = pd.to_datetime(x["Start_date"], errors="coerce", utc=True)
            q = numeric(x.get("Qc_CPC_SMPS", pd.Series(index=x.index, dtype=float)))
            for c in ["P_SMPS", "T_SMPS", "RH_SMPS"]:
                if c in x:
                    x[c] = numeric(x[c])
            x["pnsd_base_ok"] = q.isna() | (q == 0)
            x["pnsd_strict_ok"] = q == 0
            bands = pnsd_band_values(x)
            keep = ["timestamp", "pnsd_base_ok", "pnsd_strict_ok", "Qc_ACSM_SMPS"]
            keep += [c for c in ["P_SMPS", "T_SMPS", "RH_SMPS"] if c in x]
            x = pd.concat([x[keep], bands], axis=1)
            frames.append(x)
        result[site] = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        result[site] = result[site].drop_duplicates("timestamp", keep="last")
    excluded = {
        s.strip().upper()
        for s in os.environ.get("NC_PNAS_EXCLUDE_SITES", "").split(",")
        if s.strip()
    }
    return {site: frame for site, frame in result.items() if site not in excluded}
