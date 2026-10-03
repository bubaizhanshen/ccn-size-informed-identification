"""Rebuild audited 15–300 nm spectra from the external hourly archive.

No observations are bundled. Mapping/QC and the training boundary match the
manuscript; complete-cohort caches are diagnostics, not primary eligibility.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import ROOT, TEST_BOUNDARIES
import shared_pnsd_preflight as pre
import preprocessing_complete_cohort as cohort


def prepare_mapping(site):
    paths = cohort.source_smps(site)
    if not paths:
        raise FileNotFoundError(f"{site}: missing hourly SMPS archive files")
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    columns = [c for c in frames[0] if c.startswith("D_")]
    diameter = np.array([float(c[2:].replace("_", ".")) for c in columns])
    order = np.argsort(diameter)
    diameter = diameter[order]
    columns = [columns[k] for k in order]
    raw = pd.concat([f[columns + ["N_CN_SMPS_STP"]] for f in frames], ignore_index=True)
    density = raw[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    supplied = pd.to_numeric(raw.N_CN_SMPS_STP, errors="coerce").to_numpy(float)
    native_edges = pre.log10_bin_edges(diameter)
    needed = (native_edges[1:] > np.log10(15.0)) & (native_edges[:-1] < np.log10(300.0))
    valid = np.isfinite(density[:, needed]).all(1) & (density[:, needed] >= 0).all(1)
    valid &= np.isfinite(supplied) & (supplied > 0)
    valid = pre.resolve_valid(density, valid)
    ratio = (300.0 / 15.0) ** (1.0 / 24)
    grid = np.geomspace(15.0 * np.sqrt(ratio), 300.0 / np.sqrt(ratio), 24)
    report, mapped, mapped_valid = pre.audit(
        density, diameter, valid, "dndlog10dp", grid, supplied
    )
    report.update(
        metadata_gate_passed=True,
        concentration_unit="particles cm-3 per log10 diameter",
        diameter_definition="midpoint_nm",
        semantics_evidence="Harmonized archive D_* metadata: dN/dlog10Dp; independent N_CN_SMPS_STP check",
        source_sha256={p.name: cohort.sha256(p) for p in paths},
        native_width_method="log-midpoint-inferred integration cells for density",
        representation_contract={
            "input_semantics": "dndlog10dp",
            "density_log_base": 10,
            "diameter_unit": "nm",
            "timezone": "UTC",
            "averaging_duration_minutes": 60,
            "negative_value_policy": "invalid, never clipped",
            "mapped_semantics": "bin_concentration",
            "target_range_nm": [15, 300],
            "target_channels": 24,
            "instrument_metadata_source": "harmonized archive metadata and manuscript Table S1",
        },
    )
    if not report["preprocessing_gate_passed"]:
        raise ValueError(f"{site}: PNSD representation/conservation audit failed")
    dest = ROOT / "results/mapped"
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / f"{site}_mapped24.npz"
    if output.exists():
        raise FileExistsError(f"Preserve existing mapped output: {output}")
    np.savez_compressed(
        output,
        diameter_midpoint_nm=grid,
        pnsd_bin_number_concentration=mapped,
        valid_hour=mapped_valid,
        target_delta_log10_dp=pre.log10_bin_widths(grid),
    )
    (dest / f"{site}_mapped24_audit.json").write_text(json.dumps(report, indent=2))
    (dest / f"{site}_cohort_manifest.json").write_text(
        json.dumps(
            {
                "source_csv_sha256": report["source_sha256"],
                "first_test": TEST_BOUNDARIES[site],
            },
            indent=2,
        )
    )
    print(site, "PNSD gate passed", int(mapped_valid.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=cohort.SITES)
    args = parser.parse_args()
    sites = (args.site,) if args.site else cohort.SITES
    for site in sites:
        prepare_mapping(site)
    smps = cohort.ccn.harmonized.load_smps()
    ccns = cohort.load_ccn_instrument_only()
    for site in sites:
        cohort.prepare(site, smps[site], ccns[site], ROOT / "results/complete_cohort")


if __name__ == "__main__":
    main()
