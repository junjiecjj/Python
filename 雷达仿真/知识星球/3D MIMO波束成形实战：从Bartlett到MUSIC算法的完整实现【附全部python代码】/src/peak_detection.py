"""
Peak Detection Module

Extracts strongest peaks from MUSIC spectrum.
"""

import numpy as np
from config import NUM_TARGETS, AZIMUTH_SCAN, ELEVATION_SCAN


def estimate_angles(P_music):
    """
    Finds top-N peak angles from MUSIC spectrum.

    Returns:
        List of (azimuth, elevation)
    """

    flat_indices = np.argsort(P_music.ravel())[::-1]
    top_indices = flat_indices[:NUM_TARGETS]

    estimated_angles = []

    for idx in top_indices:
        el_idx, az_idx = np.unravel_index(idx, P_music.shape)

        est_az = AZIMUTH_SCAN[az_idx]
        est_el = ELEVATION_SCAN[el_idx]

        estimated_angles.append((est_az, est_el))

    return estimated_angles