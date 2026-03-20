"""
Conventional Beamforming Module

"""

import numpy as np
from config import AZIMUTH_SCAN, ELEVATION_SCAN
from signal_simulator import steering_vector
from array_geometry import generate_array_positions


def compute_bartlett_spectrum(R):
    """
    Computes Bartlett beamforming spectrum.

    Parameters:
        R : covariance matrix

    Returns:
        P_bartlett : 2D spectrum
    """

    array_positions = generate_array_positions()

    P_bartlett = np.zeros((len(ELEVATION_SCAN), len(AZIMUTH_SCAN)))

    for i, phi in enumerate(ELEVATION_SCAN):
        for j, theta in enumerate(AZIMUTH_SCAN):

            a = steering_vector(theta, phi, array_positions)

            P_bartlett[i, j] = np.real(a.conj().T @ R @ a)

    P_bartlett = 10 * np.log10(P_bartlett / np.max(P_bartlett))

    return P_bartlett