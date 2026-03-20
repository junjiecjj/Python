from signal_simulator import simulate_received_signal
from covariance import compute_covariance
from music_2d import compute_music_spectrum
from beamformer import compute_bartlett_spectrum
from peak_detection import estimate_angles
from visualization import plot_all


def main():

    print("Running 3D MIMO Beamforming Simulation...")

    # Simulate received signal
    X = simulate_received_signal()

    # Compute covariance
    R = compute_covariance(X)

    # Compute MUSIC spectrum + eigenvalues
    P_music, eigvals = compute_music_spectrum(R)

    # Compute Bartlett spectrum
    P_bartlett = compute_bartlett_spectrum(R)

    # Estimate angles
    estimated_angles = estimate_angles(P_music)

    print("\nEstimated Angles (Azimuth, Elevation):")
    for angle in estimated_angles:
        print(angle)

    # Plot everything
    plot_all(P_music, P_bartlett, eigvals, estimated_angles)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()