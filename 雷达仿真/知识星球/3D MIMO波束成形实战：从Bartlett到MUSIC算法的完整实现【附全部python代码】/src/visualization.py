import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import AZIMUTH_SCAN, ELEVATION_SCAN


def plot_all(P_music, P_bartlett, eigvals, estimated_angles):

    AZ, EL = np.meshgrid(AZIMUTH_SCAN, ELEVATION_SCAN)

    # =========================
    # 3D Comparison
    # =========================
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(AZ, EL, P_bartlett, cmap='jet')
    ax1.set_title("Bartlett Beamformer")
    ax1.set_xlabel("Azimuth (deg)")
    ax1.set_ylabel("Elevation (deg)")
    ax1.set_zlabel("Spectrum (dB)")
    ax1.view_init(elev=35, azim=45)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(AZ, EL, P_music, cmap='jet')
    for az, el in estimated_angles:
        ax2.scatter(az, el, np.max(P_music), color='black', s=80)
    ax2.set_title("MUSIC Spectrum")
    ax2.set_xlabel("Azimuth (deg)")
    ax2.set_ylabel("Elevation (deg)")
    ax2.set_zlabel("Spectrum (dB)")
    ax2.view_init(elev=35, azim=45)

    plt.tight_layout()
    plt.show()

    # =========================
    # Top View Heatmap
    # =========================
    plt.figure(figsize=(8, 6))
    plt.imshow(
        P_music,
        extent=[AZIMUTH_SCAN[0], AZIMUTH_SCAN[-1],
                ELEVATION_SCAN[0], ELEVATION_SCAN[-1]],
        origin='lower',
        aspect='auto',
        cmap='turbo'
    )

    for az, el in estimated_angles:
        plt.scatter(az, el, c='white', s=120, edgecolors='black')

    plt.title("Top View — MUSIC")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.colorbar(label="Spectrum (dB)")
    plt.tight_layout()
    plt.show()

    # =========================
    # Azimuth Slice
    # =========================
    mid_el = len(ELEVATION_SCAN) // 2

    plt.figure(figsize=(8, 5))
    plt.plot(AZIMUTH_SCAN, P_music[mid_el, :], linewidth=2)
    plt.title("Azimuth Slice at Mid Elevation")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Spectrum (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =========================
    # Eigenvalue Spectrum
    # =========================
    plt.figure(figsize=(7, 5))
    plt.stem(np.abs(eigvals))
    plt.title("Eigenvalue Spectrum (Signal vs Noise Subspace)")
    plt.xlabel("Index")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()