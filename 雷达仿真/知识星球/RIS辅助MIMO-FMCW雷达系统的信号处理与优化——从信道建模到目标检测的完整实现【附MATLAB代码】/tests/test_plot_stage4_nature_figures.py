import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_stage4_nature_figures.py"
SPEC = importlib.util.spec_from_file_location("plot_stage4_nature_figures", SCRIPT_PATH)
PLOT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PLOT_MODULE)


class Stage4NatureFigureTests(unittest.TestCase):
    def test_prepare_3d_rd_preserves_noise_floor_as_finite_surface(self):
        rd_db = np.array(
            [
                [-78.0, -76.0, -74.0],
                [-72.0, -20.0, -70.0],
                [-68.0, -66.0, -64.0],
            ]
        )
        range_axis = np.array([0.0, 12.0, 24.0])
        velocity_axis = np.array([-2.0, 0.0, 2.0])

        rd_plot, _, _ = PLOT_MODULE.prepare_3d_rd(rd_db, range_axis, velocity_axis, -60.0)

        self.assertTrue(np.isfinite(rd_plot).all())
        self.assertGreater(float(rd_plot.min()), -78.0)
        self.assertLess(float(rd_plot.min()), -60.0)
        self.assertEqual(float(rd_plot.max()), -20.0)

    def test_height_layer_norm_spans_low_and_high_surface_levels(self):
        values = np.array([-66.0, -44.0, -18.0])
        normed = PLOT_MODULE.normalize_3d_heights(values, (-68.0, -16.0))

        self.assertLess(float(normed[0]), 0.1)
        self.assertGreater(float(normed[-1]), 0.9)
        self.assertGreater(float(normed[1]), float(normed[0]))


if __name__ == "__main__":
    unittest.main()
