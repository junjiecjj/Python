"""Numerical regressions for alignment, noise scaling, Doppler, and CFAR."""

import unittest

import numpy as np
from scipy.signal import correlate

import radar_utils as r
from part1_signal_modeling import marcum_q1
from part3_pulse_compression import main_lobe_width_m, peak_sidelobe_db
from part4_range_doppler import build_rd_map, delta_v
from part5_ca_cfar import simulate


class RadarTests(unittest.TestCase):
    def test_linear_filter_matches_direct_correlation(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=40) + 1j * rng.normal(size=40)
        ref = rng.normal(size=9) + 1j * rng.normal(size=9)
        expected = correlate(data, ref, mode="valid", method="direct")
        np.testing.assert_allclose(r.matched_filter(data, ref), expected, atol=1e-12)
        np.testing.assert_allclose(r.matched_filter(np.stack([data, 2*data]), ref),
                                   np.stack([expected, 2*expected]), atol=1e-12)

    def test_range_alignment_across_scene(self):
        # Includes a full echo near each observation boundary; no wraparound.
        for R_target in [3000, 12345, 35000, 71970]:
            with self.subTest(R_target=R_target):
                y = r.matched_filter(r.make_echo(R_target=R_target), r.s_tx_pulse)
                R_est = np.argmax(np.abs(y)) * r.c / (2*r.fs)
                self.assertLessEqual(abs(R_est - R_target), r.c/(4*r.fs) + 1)

    def test_sampled_snr_matches_radar_equation(self):
        # Align to an exact sample and turn Doppler off to isolate normalization.
        R_target = 35002.5
        y = r.matched_filter(r.make_echo(R_target=R_target, vr_target=0), r.s_tx_pulse)
        sampled = np.max(np.abs(y)**2) / (r.Pn * np.sum(np.abs(r.s_tx_pulse)**2))
        Pr = r.Pt*r.G**2*r.lambda_c**2*r.sigma / ((4*np.pi)**3*R_target**4*r.L)
        theory = Pr*r.tau_p / (r.k_B*r.T0*r.F)
        self.assertLess(abs(10*np.log10(sampled/theory)), 0.03)

    def test_noise_and_weibull_power(self):
        rng = np.random.default_rng(42)
        noise = r.make_noise(rng, 200_000)
        self.assertLess(abs(np.mean(np.abs(noise)**2)/r.Pn - 1), 0.02)
        for shape in [1.0, 1.5, 2.0, 3.0]:
            clutter = r.make_clutter(rng, 200_000, c_w=shape)
            self.assertLess(abs(np.mean(np.abs(clutter)**2)/r.Pc - 1), 0.03)

    def test_hann_tradeoff_and_first_sidelobe(self):
        echo = r.make_echo(vr_target=0, R_target=35002.5)
        rect = np.abs(r.matched_filter(echo, r.s_tx_pulse))
        hann = np.abs(r.matched_filter(echo, r.s_tx_pulse*np.hanning(r.N_pulse)))
        rect /= rect.max()
        hann /= hann.max()
        axis = np.arange(rect.size)*r.c/(2*r.fs)/1e3
        self.assertGreater(main_lobe_width_m(hann, axis), main_lobe_width_m(rect, axis))
        self.assertTrue(-15 < peak_sidelobe_db(rect) < -12)
        self.assertLess(peak_sidelobe_db(hann), -28)

    def test_doppler_and_aliasing(self):
        for velocity in [20.0, -30.0]:
            RD, vel, ranges = build_rd_map(velocity)
            i, j = np.unravel_index(np.argmax(RD), RD.shape)
            expected = ((2*velocity/r.lambda_c + r.PRF/2) % r.PRF - r.PRF/2)*r.lambda_c/2
            self.assertLessEqual(abs(vel[i]-expected), delta_v/2)
            self.assertLessEqual(abs(ranges[j]-r.R0), r.c/(2*r.fs))

    def test_cfar_matches_explicit_reference_windows(self):
        power = np.random.default_rng(3).exponential(size=150)
        N_ref, N_guard, Pfa = 7, 3, 1e-3
        threshold, detections = r.ca_cfar(power, N_ref, N_guard, Pfa)
        alpha = 2*N_ref*(Pfa**(-1/(2*N_ref))-1)
        expected = np.full(power.size, np.nan)
        for i in range(N_ref+N_guard, power.size-N_ref-N_guard):
            train = np.r_[power[i-N_guard-N_ref:i-N_guard], power[i+N_guard+1:i+N_guard+N_ref+1]]
            expected[i] = alpha*np.mean(train)
        np.testing.assert_allclose(threshold, expected, atol=1e-12)
        np.testing.assert_array_equal(detections, power > expected)

    def test_cfar_no_detection_and_short_input(self):
        for power in [np.zeros(100), np.ones(100), np.zeros(10)]:
            threshold, detections = r.ca_cfar(power)
            self.assertFalse(detections.any())
        self.assertTrue(np.isnan(r.ca_cfar(np.zeros(10))[0]).all())

    def test_cfar_rejects_invalid_inputs(self):
        for power, kwargs in [(np.array([-1.]), {}), (np.array([np.nan]), {}),
                              (np.zeros(100), {"N_ref": 0}), (np.zeros(100), {"N_guard": -1}),
                              (np.zeros(100), {"Pfa": 1})]:
            with self.assertRaises(ValueError):
                r.ca_cfar(power, **kwargs)

    def test_cfar_iid_false_alarm_baseline(self):
        # At 1e-3 there are enough false alarms for a useful fast regression.
        # This does NOT calibrate the correlated radar chain at nominal 1e-6.
        power = np.random.default_rng(11).exponential(size=1_000_000)
        threshold, detections = r.ca_cfar(power, Pfa=1e-3)
        measured = np.mean(detections[np.isfinite(threshold)])
        self.assertLess(abs(measured/1e-3 - 1), 0.20)

    def test_target_detection_and_absent_target(self):
        result = simulate()
        self.assertIsNotNone(result["best_target"])
        R_est = result["range_axis"][result["best_target"]]
        self.assertLessEqual(abs(R_est-r.R0), r.c/(2*r.fs))
        self.assertGreater(result["margin_dB_at_target"], 0)
        result = simulate(sigma_target=0)
        self.assertIsNone(result["best_target"])
        self.assertIsNone(result["margin_dB_at_target"])

    def test_detection_probability_limits(self):
        b = np.sqrt(-2*np.log(r.Pfa))
        self.assertAlmostEqual(float(marcum_q1(0, b)), r.Pfa, places=12)
        self.assertGreater(float(marcum_q1(np.sqrt(2*1e3), b)), 0.999)


if __name__ == "__main__":
    unittest.main()
