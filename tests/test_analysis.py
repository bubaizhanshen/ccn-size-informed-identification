"""Generated-in-memory checks; no test observations or stored datasets."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

from analysis.diagnostics import size_fraction as size
from analysis.common import pnsd as pre
from analysis.common.files import file_info
from analysis.common.runtime import WORK
from analysis.models import tabular
from analysis.primary import hourly, cross_site as blocks
from analysis.robustness import month_selection


class AnalysisChecks(unittest.TestCase):
    def test_cutoff_overlap(self):
        edges = np.linspace(np.log10(15), np.log10(300), 25)
        grid = 10 ** ((edges[:-1] + edges[1:]) / 2)
        # Uniform dN/dlogDp integrated onto bins has an analytic fraction.
        bins = np.tile(np.diff(edges), (3, 1))
        count, fraction = size.fraction_above(bins, grid, 82.0)
        np.testing.assert_allclose(count, np.log10(300 / 82), rtol=1e-12)
        np.testing.assert_allclose(
            fraction, np.log10(300 / 82) / np.log10(300 / 15), rtol=1e-12
        )

    def test_grid_resolution(self):
        self.assertTrue(pre.synthetic_resolution_check()["passed"])

    def test_natural_log_density_conversion(self):
        diameter = np.geomspace(15, 300, 24)
        density = np.full((3, 24), 100.0)
        np.testing.assert_allclose(
            pre.density_log10(density / np.log(10), diameter, "dndln_dp"),
            density,
        )

    def test_bin_concentration_needs_widths(self):
        diameter = np.geomspace(15, 300, 24)
        with self.assertRaises(ValueError):
            pre.density_log10(np.ones((3, 24)), diameter, "bin_concentration")

    def test_duplicate_diameters_rejected(self):
        with self.assertRaises(ValueError):
            pre.log10_bin_edges(np.array([15.0, 30.0, 30.0, 100.0]))

    def test_uniform_density_conservation(self):
        edges = np.linspace(np.log10(15), np.log10(300), 25)
        grid = 10 ** ((edges[:-1] + edges[1:]) / 2)
        density = np.full((3, 24), 100.0)
        total = np.full(3, 100.0 * np.log10(300 / 15))
        report, mapped, valid = pre.audit(
            density, grid, np.ones(3, bool), "dndlog10dp", grid, total
        )
        self.assertTrue(report["preprocessing_gate_passed"])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(mapped.sum(axis=1), total)

    def test_tabular_representation_shapes(self):
        d = {
            "x_N": np.ones((10, 6)),
            "x_N82": np.ones((10, 6)),
            "x_f82": np.full((10, 6), 0.5),
            "x_fraction24": np.full((10, 6, 24), 1 / 24),
            "calendar": np.zeros((10, 4)),
        }
        self.assertEqual(tabular.features(d, "N_plus_f82", "current").shape, (10, 6))
        self.assertEqual(tabular.features(d, "N_plus_f82", "history6").shape, (10, 16))
        self.assertEqual(tabular.features(d, "full24", "current").shape, (10, 29))

    def test_within_month_selection(self):
        probabilities = np.ones(20)
        months = np.repeat(["2020-01", "2020-02"], 10)
        selected = month_selection.select(probabilities, months, "within_year_month")
        np.testing.assert_array_equal(np.flatnonzero(selected), [0, 10])

    def test_file_description(self):
        with TemporaryDirectory(dir=WORK) as directory:
            path = Path(directory) / "input.txt"
            path.touch()
            self.assertEqual(file_info(path), {"name": "input.txt", "size_bytes": 0})

    def test_label_end_purge(self):
        ts = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
        cut = ts[70]
        fit, test = hourly._fit_masks(ts, np.ones(len(ts), bool), cut, 6)
        self.assertTrue(np.all(ts[fit] + pd.Timedelta(hours=6) < cut))
        self.assertTrue(np.all(ts[test] >= cut))
        self.assertFalse(np.any(fit & test))

    def test_stable_selection(self):
        p = np.ones(23)
        np.testing.assert_array_equal(blocks.rank(p), np.array([0, 1, 2]))

    def test_bootstrap_identical_predictions(self):
        y = np.linspace(0, 2, 200)
        dates = np.repeat(np.arange(20).astype(str), 10)
        delta, lo, hi = size.paired_delta(y, y + 0.1, y + 0.1, dates, 42)
        np.testing.assert_allclose([delta, lo, hi], 0)


if __name__ == "__main__":
    unittest.main()
