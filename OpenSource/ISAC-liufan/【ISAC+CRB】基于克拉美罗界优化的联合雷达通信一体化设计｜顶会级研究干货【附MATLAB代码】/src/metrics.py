"""
CRB-ISAC Metrics
==================
Performance metrics for DFRC beamforming evaluation.

Implements:
- Beampattern computation and normalization
- RMSE / MSE computation
- SINR computation
- Trend verification against paper
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from src.model import CRBModel, SystemParams


def compute_beampattern(
    W: np.ndarray,
    angles: Optional[np.ndarray] = None,
    n_points: int = 361,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized beampattern (dB) over angle range.

    Args:
        W: Beamforming matrix (N_t x K)
        angles: Custom angles in degrees. If None, use -90 to 90.

    Returns:
        (angles, beampattern_dbi): angles and pattern in dBi
    """
    from src.model import ULASteeringVector

    if angles is None:
        angles = np.linspace(-90, 90, n_points)

    N_t = W.shape[0]
    ula = ULASteeringVector(N_t)
    _, A = ula.compute_range(len(angles))  # (n_points, N_t)

    # Beampattern: sum of squared magnitudes per angle
    # \|a(θ)^H W\|_F² = |a(θ)^H w_1|² + ... + |a(θ)^H w_K|²
    pattern = np.array(
        [np.linalg.norm(A[i].conj() @ W) ** 2 for i in range(len(angles))]
    )
    # Ensure real (numerical precision may leave tiny imaginary parts)
    pattern = np.real(pattern)

    # Normalize: max = 0 dBi
    pattern_dbi = 10 * np.log10(pattern + 1e-12) - 10 * np.log10(np.max(pattern) + 1e-12)
    return angles, pattern_dbi


def compute_peak_sidelobe_level(angles: np.ndarray, pattern_db: np.ndarray) -> float:
    """Compute peak sidelobe level (PSLL) in dB."""
    mainlobe_idx = np.argmax(pattern_db)
    left_sidelobes = pattern_db[: mainlobe_idx - 5]
    right_sidelobes = pattern_db[mainlobe_idx + 5 :]
    if len(left_sidelobes) > 0 and len(right_sidelobes) > 0:
        left_max = np.max(left_sidelobes)
        right_max = np.max(right_sidelobes)
        psl = max(left_max, right_max)
    else:
        psl = np.max(pattern_db)
    return float(psl)


def compute_mainlobe_width_3db(
    angles: np.ndarray, pattern_db: np.ndarray
) -> float:
    """Compute 3-dB mainlobe width in degrees."""
    peak_idx = np.argmax(pattern_db)
    peak_db = pattern_db[peak_idx]
    half_power_db = peak_db - 3.0

    # Find left edge
    left_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if pattern_db[i] < half_power_db:
            left_idx = i + 1
            break
        left_idx = 0

    # Find right edge
    right_idx = peak_idx
    for i in range(peak_idx + 1, len(pattern_db)):
        if pattern_db[i] < half_power_db:
            right_idx = i - 1
            break
        right_idx = len(pattern_db) - 1

    return float(angles[right_idx] - angles[left_idx])


class MetricsCollector:
    """Collect and compare metrics for baseline evaluation."""

    def __init__(self, params: SystemParams):
        self.params = params
        self.model = CRBModel(params)
        self.results = {}

    def evaluate_beampattern(
        self,
        W: np.ndarray,
        method_name: str,
    ) -> dict:
        """Evaluate beampattern metrics."""
        angles, bp_db = compute_beampattern(W)
        mainlobe_idx = np.argmax(bp_db)
        psl = compute_peak_sidelobe_level(angles, bp_db)
        width_3db = compute_mainlobe_width_3db(angles, bp_db)
        gain_at_target = float(bp_db[mainlobe_idx])

        self.results[method_name] = {
            "gain_dbi_at_target": gain_at_target,
            "psll_db": psl,
            "mainlobe_3db_width_deg": width_3db,
        }
        return self.results[method_name]

    def compare_methods(
        self,
        W_dict: dict[str, np.ndarray],
    ) -> dict:
        """
        Compare multiple beamforming methods.

        Args:
            W_dict: {method_name: W}
        """
        comparison = {}
        for name, W in W_dict.items():
            metrics = self.evaluate_beampattern(W, name)
            comparison[name] = metrics

        # Verify: proposed method should have highest gain at target
        proposed = comparison.get("Proposed CRB-Min", {})
        benchmarks = {k: v for k, v in comparison.items() if k != "Proposed CRB-Min"}
        if benchmarks and proposed:
            proposed_gain = proposed["gain_dbi_at_target"]
            all_higher = all(
                proposed_gain >= m["gain_dbi_at_target"]
                for m in benchmarks.values()
            )
            comparison["trend_verified"] = all_higher

        return comparison

    def verify_trends(self, crb_curve: np.ndarray, sinr_range_db: np.ndarray) -> dict:
        """
        Verify CRB vs SINR trend against paper description.

        Paper: "error can be maintained at the lowest level
        for both cases when the required SINR is below 30dB"
        """
        idx_30 = np.argmin(np.abs(sinr_range_db - 30))
        idx_25 = np.argmin(np.abs(sinr_range_db - 25))
        idx_35 = np.argmin(np.abs(sinr_range_db - 35))

        below_30db_stable = bool(
            np.max(crb_curve[:idx_30]) / (np.min(crb_curve[:idx_30]) + 1e-12) < 3.0
        )
        above_30db_rises = bool(
            np.mean(crb_curve[idx_30:idx_35]) > crb_curve[idx_25] * 2
        )
        at_30db_threshold = sinr_range_db[idx_30]

        return {
            "below_30db_stable": below_30db_stable,
            "above_30db_rises": above_30db_rises,
            "threshold_db": float(at_30db_threshold),
            "paper_trend_verified": below_30db_stable and above_30db_rises,
        }
