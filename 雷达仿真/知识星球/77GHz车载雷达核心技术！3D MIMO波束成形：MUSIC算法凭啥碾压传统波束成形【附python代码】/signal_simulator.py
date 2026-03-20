"""
Signal Simulation Module

"""

import numpy as np
from config import LAMBDA, NUM_SNAPSHOTS, SNR_DB, TARGETS
from array_geometry import generate_array_positions


def steering_vector(theta_deg, phi_deg, array_positions):
    """
    Computes steering vector for given azimuth & elevation.
    """

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    kx = (2 * np.pi / LAMBDA) * np.cos(phi) * np.sin(theta)
    ky = (2 * np.pi / LAMBDA) * np.sin(phi)

    phase = kx * array_positions[0] + ky * array_positions[1]

    return np.exp(1j * phase)


def simulate_received_signal():
    """
    Simulates received array data matrix X.
    """

    array_positions = generate_array_positions()
    num_elements = array_positions.shape[1]

    X = np.zeros((num_elements, NUM_SNAPSHOTS), dtype=complex)

    for theta, phi in TARGETS:
        a = steering_vector(theta, phi, array_positions)
        signal = (np.random.randn(1, NUM_SNAPSHOTS) +
                  1j * np.random.randn(1, NUM_SNAPSHOTS))
        X += a[:, None] @ signal

    noise = (np.random.randn(num_elements, NUM_SNAPSHOTS) +
             1j * np.random.randn(num_elements, NUM_SNAPSHOTS)) / np.sqrt(2)

    X += noise * 10 ** (-SNR_DB / 20)

    return X