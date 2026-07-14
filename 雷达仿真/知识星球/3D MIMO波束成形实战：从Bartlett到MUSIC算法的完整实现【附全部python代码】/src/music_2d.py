import numpy as np
from numpy.linalg import eig
from config import NUM_TARGETS, AZIMUTH_SCAN, ELEVATION_SCAN
from signal_simulator import steering_vector
from array_geometry import generate_array_positions


def compute_music_spectrum(R):

    # Eigen decomposition
    eigvals, eigvecs = eig(R)

    # Sort descending
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Noise subspace
    En = eigvecs[:, NUM_TARGETS:]

    array_positions = generate_array_positions()

    P_music = np.zeros((len(ELEVATION_SCAN), len(AZIMUTH_SCAN)))

    for i, phi in enumerate(ELEVATION_SCAN):
        for j, theta in enumerate(AZIMUTH_SCAN):
            a = steering_vector(theta, phi, array_positions)
            denom = a.conj().T @ En @ En.conj().T @ a
            P_music[i, j] = 1 / np.abs(denom)

    # Normalize
    P_music = 10 * np.log10(P_music / np.max(P_music))

    # IMPORTANT: return BOTH values
    return P_music, eigvals