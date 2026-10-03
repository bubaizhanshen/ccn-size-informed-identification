"""Generated-in-memory checks; no test observations or stored datasets."""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import diagnostic_size_fraction as size
import shared_pnsd_preflight as pre
import primary_hourly as hourly
import primary_cross_site as blocks


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
