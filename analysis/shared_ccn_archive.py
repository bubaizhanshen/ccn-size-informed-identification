"""Shared CCN archive covariate definitions (no analysis launcher)."""

import numpy as np
import pandas as pd
import shared_archive_features as harmonized

BASE = ["P_SMPS", "T_SMPS", "RH_SMPS", "sin_hour", "cos_hour", "sin_year", "cos_year"]
SPECTRUM = [
    "n_15_25",
    "n_25_50",
    "n_50_100",
    "n_100_300",
    "n_15_300",
    "frac_15_25",
    "frac_25_50",
    "frac_50_100",
    "frac_100_300",
    "pnsd_gmd",
]


def make_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = df[cols].copy()
    for c in cols:
        if c.startswith("n_") or c in {"pnsd_gmd", "N_CN_CPC_STP_B"}:
            z[c] = np.log1p(z[c].clip(lower=0))
    return z
