import numpy as np
import scipy
from configs.isac import fc, fs, B, T_prbs, T_d
from configs.ml import domain


def range_doppler_response(data, domain=domain):
    """
    Compute Range-Doppler response for a radar signal.

    Parameters:
    -----------
    data : ndarray
        2D input signal array (n_pulses, n_samples_per_pulse).
    fs : float
        Sample rate of the signal (Hz).
    B : float
        Bandwidth of the chirp (Hz).
    T_prbs : float
        Pulse duration (seconds).
    fc : float
        Operating (carrier) frequency (Hz).
    T_d : float
        Pulse repetition interval (seconds).

    Returns:
    --------
    rd_map : ndarray
        Range-Doppler map (range bins x Doppler bins).
    range_axis : ndarray
        Range axis (meters).
    speed_axis : ndarray
        Doppler speed axis (m/s).
    """
    # Ensure data is a NumPy array
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            "Input data must be a 2D array (n_pulses, n_samples_per_pulse)"
        )

    # Calculate parameters
    sweep_slope = -B / T_prbs  # Negative for downchirp (Hz/s)
    prf = 1 / T_d  # Pulse repetition frequency (Hz)

    # Time domain given y(n)
    if domain == "time":
        rd_map = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)

    # Frequency domain given Y(f)
    elif domain == "freq":
        doppler_fft = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)
        rd_map = np.fft.ifft(doppler_fft, axis=0)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Extract dimensions
    N_chip, N_sym = data.shape

    # Compute range axis (non-centered, starting at 0)
    range_resolution = scipy.constants.speed_of_light / (2 * B)  # Range resolution (m)
    max_range = range_resolution * N_chip  # Max unambiguous range
    range_axis = np.linspace(0, max_range, N_chip)

    # Compute Doppler speed axis
    max_doppler_freq = prf / 2  # Max unambiguous Doppler frequency (Nyquist)
    doppler_freq = np.linspace(-max_doppler_freq, max_doppler_freq, N_sym)
    # Convert Doppler frequency to speed: speed = (doppler_freq * c) / (2 * fc)
    speed_axis = (doppler_freq * scipy.constants.speed_of_light) / (2 * fc)
    # Convert to dB for visualization
    rd_map = np.log10(np.abs(rd_map) + 1e-10)  # Avoid log(0)
    # rd_map = np.abs(rd_map)

    return rd_map, range_axis, speed_axis
