"""Validate PNSD semantics and perform bin-width-aware grid mapping."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEMANTICS = ("bin_concentration", "dndlog10dp", "dndln_dp")
ALLOWED_WIDTH_METHODS = (
    "instrument_documented_bin_edges",
    "source_processing_code_conversion_weights",
    "source_density_to_bin_conversion_delta_log10_dp",
    "source_forward_spacing_scale_crosschecked_delta_log10_dp",
)
MEDIAN_TOLERANCE = 0.03
TAIL_TOLERANCE = 0.05
TOTAL_TOLERANCE = 0.05
MIN_MAPPED_VALID_FRACTION = 0.95


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_value(data: object, key: str) -> object | None:
    if key not in data:
        return None
    values = np.asarray(data[key])
    if values.size != 1:
        raise ValueError(f"Embedded field {key!r} must be scalar")
    value = values.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if hasattr(value, "item") else value


def log10_bin_edges(diameters: np.ndarray) -> np.ndarray:
    diameters = np.asarray(diameters, dtype=float)
    if (
        diameters.ndim != 1
        or len(diameters) < 2
        or (not np.isfinite(diameters).all())
        or np.any(diameters <= 0)
        or np.any(np.diff(diameters) <= 0)
    ):
        raise ValueError("diameters must be finite, positive, unique, and increasing")
    centers = np.log10(diameters)
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def log10_bin_widths(diameters: np.ndarray) -> np.ndarray:
    return np.diff(log10_bin_edges(diameters))


def as_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[None, :]
    if values.ndim != 2:
        raise ValueError("spectrum array must be one- or two-dimensional")
    return values


def density_log10(
    spectra: np.ndarray,
    diameters: np.ndarray,
    semantics: str,
    native_widths: np.ndarray | None = None,
) -> np.ndarray:
    spectra = as_matrix(spectra)
    if semantics == "bin_concentration":
        if native_widths is None:
            raise ValueError(
                "Per-bin concentration requires an explicit native delta-log10-Dp vector; midpoint-only inference is blocked"
            )
        native_widths = np.asarray(native_widths, dtype=float)
        if native_widths.shape != np.asarray(diameters).shape:
            raise ValueError("native width vector shape differs from diameter axis")
        return spectra / native_widths[None, :]
    if semantics == "dndlog10dp":
        return spectra
    if semantics == "dndln_dp":
        return spectra * np.log(10.0)
    raise ValueError(f"Unsupported semantics: {semantics}")


def integrate_native_range(
    spectra: np.ndarray,
    diameters: np.ndarray,
    semantics: str,
    lower_log10: float,
    upper_log10: float,
    native_widths: np.ndarray | None = None,
) -> np.ndarray:
    spectra = as_matrix(spectra)
    if semantics == "bin_concentration":
        if native_widths is None:
            raise ValueError(
                "Per-bin concentration integration requires explicit native conversion widths"
            )
        widths = np.asarray(native_widths, dtype=float)
        starts = np.log10(np.asarray(diameters, dtype=float))
        ends = starts + widths
        overlap = np.maximum(
            0.0, np.minimum(ends, upper_log10) - np.maximum(starts, lower_log10)
        )
        fraction = np.divide(
            overlap,
            widths,
            out=np.zeros_like(overlap),
            where=np.isfinite(widths) & (widths > 0),
        )
        contribution = spectra * fraction[None, :]
    else:
        edges = log10_bin_edges(diameters)
        overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], upper_log10) - np.maximum(edges[:-1], lower_log10),
        )
        contribution = (
            density_log10(spectra, diameters, semantics, native_widths)
            * overlap[None, :]
        )
    return np.nansum(contribution, axis=1)


def full_native_integral(
    spectra: np.ndarray,
    diameters: np.ndarray,
    semantics: str,
    native_widths: np.ndarray | None = None,
) -> np.ndarray:
    spectra = as_matrix(spectra)
    if semantics == "bin_concentration":
        return np.nansum(spectra, axis=1)
    return np.nansum(
        density_log10(spectra, diameters, semantics, native_widths)
        * (
            np.asarray(native_widths, dtype=float)[None, :]
            if native_widths is not None
            else log10_bin_widths(diameters)[None, :]
        ),
        axis=1,
    )


def harmonize_to_bin_concentration(
    spectra: np.ndarray,
    diameters: np.ndarray,
    valid: np.ndarray,
    semantics: str,
    grid: np.ndarray,
    native_widths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    spectra = as_matrix(spectra)
    diameters = np.asarray(diameters, dtype=float)
    grid = np.asarray(grid, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if spectra.shape != (len(valid), len(diameters)):
        raise ValueError("spectrum shape is inconsistent with metadata")
    if grid[0] < diameters[0] or grid[-1] > diameters[-1]:
        raise ValueError("target grid lies outside the native midpoint range")
    source_density = density_log10(spectra, diameters, semantics, native_widths)
    target_width = log10_bin_widths(grid)
    log_native = np.log(diameters)
    log_grid = np.log(grid)
    mapped = np.full((len(spectra), len(grid)), np.nan, dtype=np.float64)
    for row_index in np.flatnonzero(valid):
        row = source_density[row_index]
        finite = np.isfinite(row) & (row >= 0)
        if finite.sum() < 2:
            continue
        finite_diameter = diameters[finite]
        if finite_diameter[0] > grid[0] or finite_diameter[-1] < grid[-1]:
            continue
        mapped[row_index] = (
            np.interp(log_grid, log_native[finite], row[finite]) * target_width
        )
    mapped_valid = np.isfinite(mapped).all(axis=1)
    return (mapped, mapped_valid)


def synthetic_resolution_check() -> dict[str, float | bool]:
    """Verify that source channel count does not set mapped amplitude."""
    coarse = np.geomspace(10.0, 100.0, 9)
    dense = np.geomspace(10.0, 100.0, 33)
    target = np.geomspace(20.0, 50.0, 7)
    density = 1000.0
    coarse_spectra = (density * log10_bin_widths(coarse))[None, :]
    dense_spectra = (density * log10_bin_widths(dense))[None, :]
    coarse_mapped, coarse_valid = harmonize_to_bin_concentration(
        coarse_spectra,
        coarse,
        np.array([True]),
        "bin_concentration",
        target,
        log10_bin_widths(coarse),
    )
    dense_mapped, dense_valid = harmonize_to_bin_concentration(
        dense_spectra,
        dense,
        np.array([True]),
        "bin_concentration",
        target,
        log10_bin_widths(dense),
    )
    relative = np.max(
        np.abs(coarse_mapped - dense_mapped) / np.maximum(np.abs(dense_mapped), 1e-12)
    )
    passed = bool(coarse_valid[0] and dense_valid[0] and (relative <= 1e-10))
    return {"maximum_relative_difference": float(relative), "passed": passed}


def finite_quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"q05": float("nan"), "median": float("nan"), "q95": float("nan")}
    q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {"q05": float(q05), "median": float(median), "q95": float(q95)}


def gate_interval(
    statistics: dict[str, float], median_tolerance: float, tail_tolerance: float
) -> bool:
    return bool(
        np.isfinite(statistics["median"])
        and abs(statistics["median"] - 1.0) <= median_tolerance
        and (statistics["q05"] >= 1.0 - tail_tolerance)
        and (statistics["q95"] <= 1.0 + tail_tolerance)
    )


def resolve_valid(spectra: np.ndarray, supplied: np.ndarray | None) -> np.ndarray:
    spectra = as_matrix(spectra)
    finite_fraction = np.isfinite(spectra).mean(axis=1)
    positive = np.nansum(np.where(np.isfinite(spectra), spectra, 0.0), axis=1) > 0
    valid = (finite_fraction >= 0.8) & positive
    if supplied is not None:
        supplied = np.asarray(supplied).reshape(-1)
        if len(supplied) != len(spectra):
            raise ValueError("valid-key length differs from spectra rows")
        valid &= supplied.astype(bool)
    return valid


def audit(
    spectra: np.ndarray,
    diameters: np.ndarray,
    valid: np.ndarray,
    semantics: str,
    grid: np.ndarray,
    supplied_total: np.ndarray | None = None,
    native_widths: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    spectra = as_matrix(spectra)
    diameters = np.asarray(diameters, dtype=float)
    grid = np.asarray(grid, dtype=float)
    log10_bin_edges(diameters)
    log10_bin_edges(grid)
    explicit_width_gate = True
    width_summary = None
    if native_widths is not None:
        native_widths = np.asarray(native_widths, dtype=float)
        if native_widths.shape != diameters.shape:
            raise ValueError("native width vector shape differs from diameter axis")
        usable_width = np.isfinite(native_widths) & (native_widths > 0)
        width_support = diameters[usable_width]
        explicit_width_gate = bool(
            usable_width.sum() >= 2
            and width_support.min() <= grid[0]
            and (width_support.max() >= grid[-1])
        )
        width_summary = {
            "n_usable_channels": int(usable_width.sum()),
            "minimum": float(np.nanmin(native_widths)),
            "median": float(np.nanmedian(native_widths)),
            "maximum": float(np.nanmax(native_widths)),
        }
    elif semantics == "bin_concentration":
        explicit_width_gate = False
        raise ValueError(
            "Per-bin concentration preprocessing is blocked without an explicit native width vector"
        )
    mapped, mapped_valid = harmonize_to_bin_concentration(
        spectra, diameters, valid, semantics, grid, native_widths
    )
    target_widths = log10_bin_widths(grid)
    if semantics == "bin_concentration":
        target_lower_log10 = float(np.log10(grid[0]))
        target_upper_log10 = float(np.log10(grid[-1]) + target_widths[-1])
        target_interval_convention = "source_processing_forward_intervals"
    else:
        target_edges = log10_bin_edges(grid)
        target_lower_log10 = float(target_edges[0])
        target_upper_log10 = float(target_edges[-1])
        target_interval_convention = "midpoint_inferred_density_cells"
    native_target_total = integrate_native_range(
        spectra,
        diameters,
        semantics,
        target_lower_log10,
        target_upper_log10,
        native_widths,
    )
    usable = mapped_valid & np.isfinite(native_target_total) & (native_target_total > 0)
    mapped_total = np.nansum(mapped, axis=1)
    conservation = finite_quantiles(mapped_total[usable] / native_target_total[usable])
    conservation_gate = gate_interval(conservation, MEDIAN_TOLERANCE, TAIL_TOLERANCE)
    n_input_valid = int(np.asarray(valid).sum())
    mapped_valid_fraction = float(mapped_valid.sum() / max(n_input_valid, 1))
    mapped_coverage_gate = bool(mapped_valid_fraction >= MIN_MAPPED_VALID_FRACTION)
    negative_cells = int(np.sum(np.isfinite(spectra[valid]) & (spectra[valid] < 0)))
    nonnegative_gate = negative_cells == 0
    synthetic_check = synthetic_resolution_check()
    total_check = None
    total_gate = True
    if supplied_total is not None:
        supplied_total = np.asarray(supplied_total, dtype=float).reshape(-1)
        if len(supplied_total) != len(spectra):
            raise ValueError("total-key length differs from spectra rows")
        native_full = full_native_integral(spectra, diameters, semantics, native_widths)
        total_usable = (
            valid
            & np.isfinite(native_full)
            & (native_full > 0)
            & np.isfinite(supplied_total)
            & (supplied_total > 0)
        )
        total_check = finite_quantiles(
            native_full[total_usable] / supplied_total[total_usable]
        )
        total_gate = gate_interval(total_check, TOTAL_TOLERANCE, TOTAL_TOLERANCE)
    direct_bias = None
    if semantics == "bin_concentration":
        direct = np.full_like(mapped, np.nan)
        log_native = np.log(diameters)
        log_grid = np.log(grid)
        for row_index in np.flatnonzero(valid):
            row = spectra[row_index]
            finite = np.isfinite(row) & (row >= 0)
            if (
                finite.sum() >= 2
                and diameters[finite][0] <= grid[0]
                and (diameters[finite][-1] >= grid[-1])
            ):
                direct[row_index] = np.interp(log_grid, log_native[finite], row[finite])
        direct_valid = np.isfinite(direct).all(axis=1) & usable
        direct_bias = finite_quantiles(
            np.nansum(direct[direct_valid], axis=1) / native_target_total[direct_valid]
        )
    passed = bool(
        conservation_gate
        and total_gate
        and usable.any()
        and mapped_coverage_gate
        and nonnegative_gate
        and synthetic_check["passed"]
        and explicit_width_gate
    )
    report: dict[str, object] = {
        "input_semantics": semantics,
        "input_semantics_explicit": True,
        "n_rows": int(len(spectra)),
        "n_native_channels": int(len(diameters)),
        "native_lower_nm": float(diameters[0]),
        "native_upper_nm": float(diameters[-1]),
        "native_width_vector_explicit": native_widths is not None,
        "native_width_summary_log10": width_summary,
        "native_width_gate_passed": explicit_width_gate,
        "target_grid_nm": [float(value) for value in grid],
        "conservation_interval_log10": [target_lower_log10, target_upper_log10],
        "target_interval_convention": target_interval_convention,
        "n_input_valid_rows": n_input_valid,
        "n_mapped_valid_rows": int(mapped_valid.sum()),
        "mapped_valid_fraction": mapped_valid_fraction,
        "mapped_coverage_gate_passed": mapped_coverage_gate,
        "n_negative_cells_in_valid_rows": negative_cells,
        "nonnegative_gate_passed": nonnegative_gate,
        "synthetic_resolution_check": synthetic_check,
        "conservation_mapped_over_native": conservation,
        "conservation_gate_passed": conservation_gate,
        "supplied_total_check_available": supplied_total is not None,
        "native_integral_over_supplied_total": total_check,
        "supplied_total_gate_passed": total_gate,
        "direct_bin_interpolation_over_native": direct_bias,
        "preprocessing_gate_passed": passed,
        "mapped_semantics": "bin_concentration",
        "mapped_key": "pnsd_bin_number_concentration",
    }
    return (report, mapped, mapped_valid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--diameter-key", required=True)
    parser.add_argument("--spectrum-key", required=True)
    parser.add_argument("--input-semantics", choices=SEMANTICS, required=True)
    parser.add_argument("--valid-key")
    parser.add_argument("--total-key")
    parser.add_argument("--time-key")
    parser.add_argument("--width-key")
    parser.add_argument("--width-method", choices=ALLOWED_WIDTH_METHODS)
    parser.add_argument("--width-evidence")
    parser.add_argument("--concentration-unit")
    parser.add_argument("--diameter-definition", choices=("midpoint_nm",))
    parser.add_argument("--semantics-evidence")
    parser.add_argument("--grid-lower", type=float, required=True)
    parser.add_argument("--grid-upper", type=float, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapped-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_size < 2:
        raise ValueError("grid-size must be at least two")
    with np.load(args.input) as data:
        diameters = data[args.diameter_key].astype(float)
        spectra = as_matrix(data[args.spectrum_key].astype(float))
        embedded = {
            key: scalar_value(data, key)
            for key in (
                "representation_contract_version",
                "spectrum_semantics",
                "concentration_unit",
                "diameter_definition",
                "semantics_evidence",
                "native_width_method",
                "native_width_evidence",
            )
        }
        native_widths = (
            data[args.width_key].astype(float) if args.width_key is not None else None
        )
        supplied_valid = data[args.valid_key] if args.valid_key is not None else None
        valid = resolve_valid(spectra, supplied_valid)
        supplied_total = data[args.total_key] if args.total_key is not None else None
        times = data[args.time_key] if args.time_key is not None else None
    embedded_semantics = embedded["spectrum_semantics"]
    if embedded_semantics is not None and embedded_semantics != args.input_semantics:
        raise ValueError(
            f"Declared input semantics conflict with the embedded contract: declared={args.input_semantics!r}, embedded={embedded_semantics!r}"
        )
    embedded_unit = embedded["concentration_unit"]
    if (
        args.concentration_unit is not None
        and embedded_unit is not None
        and (args.concentration_unit != embedded_unit)
    ):
        raise ValueError(
            f"Declared concentration unit conflicts with the embedded contract: declared={args.concentration_unit!r}, embedded={embedded_unit!r}"
        )
    embedded_diameter = embedded["diameter_definition"]
    if (
        args.diameter_definition is not None
        and embedded_diameter is not None
        and (args.diameter_definition != embedded_diameter)
    ):
        raise ValueError(
            f"Declared diameter definition conflicts with the embedded contract: declared={args.diameter_definition!r}, embedded={embedded_diameter!r}"
        )
    embedded_width_method = embedded["native_width_method"]
    if (
        args.width_method is not None
        and embedded_width_method is not None
        and (args.width_method != embedded_width_method)
    ):
        raise ValueError(
            f"Declared native width method conflicts with the embedded contract: declared={args.width_method!r}, embedded={embedded_width_method!r}"
        )
    embedded_width_evidence = embedded["native_width_evidence"]
    if (
        args.width_evidence is not None
        and embedded_width_evidence is not None
        and (args.width_evidence != embedded_width_evidence)
    ):
        raise ValueError(
            f"Declared native width evidence conflicts with the embedded contract: declared={args.width_evidence!r}, embedded={embedded_width_evidence!r}"
        )
    concentration_unit = args.concentration_unit or embedded_unit
    diameter_definition = args.diameter_definition or embedded_diameter
    native_width_method = args.width_method or embedded_width_method
    native_width_evidence = args.width_evidence or embedded_width_evidence
    if (
        native_width_method is not None
        and native_width_method not in ALLOWED_WIDTH_METHODS
    ):
        raise ValueError(
            f"Unsupported native width method in representation contract: {native_width_method!r}"
        )
    semantics_evidence = (
        args.semantics_evidence
        or embedded["semantics_evidence"]
        or (
            f"independent_total_key:{args.total_key}"
            if args.total_key is not None
            else None
        )
    )
    metadata_gate = bool(
        concentration_unit
        and diameter_definition == "midpoint_nm"
        and semantics_evidence
        and (
            args.input_semantics != "bin_concentration"
            or (
                native_widths is not None
                and native_width_method is not None
                and bool(native_width_evidence)
            )
        )
    )
    grid = np.geomspace(args.grid_lower, args.grid_upper, args.grid_size)
    report, mapped, mapped_valid = audit(
        spectra,
        diameters,
        valid,
        args.input_semantics,
        grid,
        supplied_total,
        native_widths,
    )
    report["input_file"] = args.input.name
    report["input_sha256"] = file_sha256(args.input)
    report["embedded_representation_contract"] = embedded
    report["concentration_unit"] = concentration_unit
    report["diameter_definition"] = diameter_definition
    report["semantics_evidence"] = semantics_evidence
    report["native_width_key"] = args.width_key
    report["native_width_method"] = native_width_method
    report["native_width_evidence"] = native_width_evidence
    report["metadata_gate_passed"] = metadata_gate
    report["preprocessing_gate_passed"] = bool(
        report["preprocessing_gate_passed"] and metadata_gate
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.mapped_output is not None:
        args.mapped_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "diameter_midpoint_nm": grid,
            "pnsd_bin_number_concentration": mapped,
            "valid_hour": mapped_valid,
            "target_delta_log10_dp": log10_bin_widths(grid),
            "native_width_method": np.asarray("explicit_source_width_to_target_grid"),
            "source_native_width_method": np.asarray(native_width_method or ""),
            "source_native_width_evidence": np.asarray(native_width_evidence or ""),
        }
        if times is not None:
            payload["datetime_utc"] = times
        np.savez_compressed(args.mapped_output, **payload)
    print(json.dumps(report, indent=2))
    if not report["preprocessing_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
