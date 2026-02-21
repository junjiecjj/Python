import numpy as np
from scipy import signal, optimize
from scipy.constants import c as speed_of_light
import warnings


def db2pow(db_val):
    """Convert dB to power"""
    return 10 ** (db_val / 10)


def pow2db(power_val):
    """Convert power to dB"""
    return 10 * np.log10(power_val)


def dop2speed(doppler_freq, wavelength):
    """Convert Doppler frequency to speed"""
    return doppler_freq * wavelength / 2


def npwgnthresh(pfa, n_samples, coherent_type="noncoherent"):
    """Noise power white Gaussian noise threshold (simplified)"""
    if coherent_type == "noncoherent":
        # Approximation for non-coherent detection
        return pow2db(-2 * np.log(pfa))
    else:
        # Coherent detection
        return pow2db(-2 * np.log(pfa))


def findpeaks_2d(data, threshold=None):
    """Find 2D peaks in data"""
    if threshold is None:
        threshold = 0

    # Simple peak finding - can be replaced with more sophisticated methods
    peaks = []
    locations = []

    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            if (
                data[i, j] > threshold
                and data[i, j] > data[i - 1, j]
                and data[i, j] > data[i + 1, j]
                and data[i, j] > data[i, j - 1]
                and data[i, j] > data[i, j + 1]
            ):
                peaks.append(data[i, j])
                locations.append((i, j))

    return np.array(peaks), locations


# 1. 2D FFT Method
def estimate_range_doppler_2dfft(
    received_waveform, noise_power, pfa, T_prbs, num_of_targets, fs, wavelength
):
    """2D FFT Range-Doppler estimation"""

    N_chip, N_sym = received_waveform.shape
    hat_r_tars = np.zeros(num_of_targets)
    hat_v_tars = np.zeros(num_of_targets)

    # Create range bins
    fasttime = np.arange(0, T_prbs, 1 / fs)
    rangebins = speed_of_light * fasttime / 2

    # 2D FFT processing
    Z_fft = np.fft.fft2(received_waveform)
    Z_fft_shifted = np.fft.fftshift(Z_fft)
    range_doppler_map = np.abs(Z_fft_shifted) ** 2

    # Doppler frequency bins
    doppler_bins = np.fft.fftshift(np.fft.fftfreq(N_sym, T_prbs))

    # Threshold
    thresh_db = npwgnthresh(pfa, N_sym, "noncoherent")
    thresh = noise_power * db2pow(thresh_db)

    # Find peaks
    peaks, locations = findpeaks_2d(range_doppler_map, thresh)

    if len(peaks) > 0:
        # Sort peaks by magnitude
        sort_idx = np.argsort(peaks)[::-1]  # Descending order

        for k in range(min(num_of_targets, len(sort_idx))):
            range_idx, doppler_idx = locations[sort_idx[k]]

            # Adjust for fftshift
            range_idx_orig = range_idx - N_chip // 2
            if range_idx_orig < 0:
                range_idx_orig += N_chip

            if range_idx_orig < len(rangebins):
                hat_r_tars[k] = rangebins[range_idx_orig]

            doppler_freq = doppler_bins[doppler_idx]
            hat_v_tars[k] = dop2speed(doppler_freq / 4, wavelength)

    return hat_r_tars, hat_v_tars


# 2. 2D MUSIC Method
def estimate_range_doppler_2dmusic(
    received_waveform, noise_power, pfa, T_prbs, num_of_targets, fs, wavelength
):
    """2D MUSIC Range-Doppler estimation"""

    N_chip, N_sym = received_waveform.shape
    hat_r_tars = np.zeros(num_of_targets)
    hat_v_tars = np.zeros(num_of_targets)

    # Create range bins
    fasttime = np.arange(0, T_prbs, 1 / fs)
    rangebins = speed_of_light * fasttime / 2

    # Vectorize the received waveform
    z_vec = received_waveform.flatten()

    # Forward-backward averaging for single snapshot
    z_fb = np.concatenate([z_vec, np.conj(np.flipud(z_vec))])
    R = np.outer(z_fb, np.conj(z_fb)) / len(z_fb)

    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(R)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Noise subspace
    En = eigenvecs[:, num_of_targets:]

    # Search grid
    range_search = np.arange(1, N_chip + 1)
    doppler_search = np.linspace(-0.5 / T_prbs, 0.5 / T_prbs, 64)

    # MUSIC spectrum
    P_music = np.zeros((len(range_search), len(doppler_search)))

    for i, range_idx in enumerate(range_search):
        for j, doppler_freq in enumerate(doppler_search):
            # Steering vector
            range_delay = (range_idx - 1) / fs

            # Create 2D steering vector
            t_fast = np.arange(N_chip) / fs
            t_slow = np.arange(N_sym) * T_prbs

            a_range = np.exp(1j * 2 * np.pi * fs * range_delay * t_fast)
            a_doppler = np.exp(1j * 2 * np.pi * doppler_freq * t_slow)
            a = np.kron(a_doppler, a_range)

            # MUSIC spectrum
            denominator = np.real(np.conj(a).T @ (En @ np.conj(En).T) @ a)
            P_music[i, j] = 1 / max(denominator, 1e-12)  # Avoid division by zero

    # Find peaks
    peaks, locations = findpeaks_2d(P_music)

    if len(peaks) > 0:
        sort_idx = np.argsort(peaks)[::-1]

        for k in range(min(num_of_targets, len(sort_idx))):
            range_idx, doppler_idx = locations[sort_idx[k]]

            if range_idx < len(rangebins):
                hat_r_tars[k] = rangebins[range_search[range_idx] - 1]

            doppler_freq = doppler_search[doppler_idx]
            hat_v_tars[k] = dop2speed(doppler_freq / 4, wavelength)

    return hat_r_tars, hat_v_tars


# 3. CLEAN Method
def estimate_range_doppler_clean(
    received_waveform, noise_power, pfa, T_prbs, num_of_targets, fs, wavelength
):
    """CLEAN Range-Doppler estimation"""

    N_chip, N_sym = received_waveform.shape
    hat_r_tars = np.zeros(num_of_targets)
    hat_v_tars = np.zeros(num_of_targets)

    # Create range bins
    fasttime = np.arange(0, T_prbs, 1 / fs)
    rangebins = speed_of_light * fasttime / 2

    # Initialize
    residual = received_waveform.copy()
    loop_gain = 0.1
    max_iterations = 100

    # Threshold
    thresh_db = npwgnthresh(pfa, N_sym, "noncoherent")
    thresh = np.sqrt(noise_power * db2pow(thresh_db))

    clean_components = []

    for iteration in range(max_iterations):
        # Find peak in residual
        max_val = np.max(np.abs(residual))
        max_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)

        if max_val < thresh or len(clean_components) >= num_of_targets:
            break

        range_idx, sym_idx = max_idx

        # Store clean component
        clean_components.append((range_idx, sym_idx, residual[max_idx]))

        # Create point spread function (simplified as delta function)
        psf = np.zeros_like(residual)
        psf[range_idx, sym_idx] = 1

        # Subtract from residual
        residual = residual - loop_gain * residual[max_idx] * psf

    # Extract targets
    for k in range(min(num_of_targets, len(clean_components))):
        range_idx, sym_idx, _ = clean_components[k]

        if range_idx < len(rangebins):
            hat_r_tars[k] = rangebins[range_idx]

        # Estimate Doppler from symbol index
        doppler_freq = (sym_idx - N_sym / 2) / (N_sym * T_prbs)
        hat_v_tars[k] = dop2speed(doppler_freq / 4, wavelength)

    return hat_r_tars, hat_v_tars


# 4. MLE Method
def estimate_range_doppler_mle(
    received_waveform, noise_power, pfa, T_prbs, num_of_targets, fs, wavelength
):
    """MLE Range-Doppler estimation"""

    N_chip, N_sym = received_waveform.shape
    hat_r_tars = np.zeros(num_of_targets)
    hat_v_tars = np.zeros(num_of_targets)

    # Create range bins
    fasttime = np.arange(0, T_prbs, 1 / fs)
    rangebins = speed_of_light * fasttime / 2

    # Time grids
    t_fast = np.arange(N_chip) / fs
    t_slow = np.arange(N_sym) * T_prbs

    # Initial estimates using 2D FFT
    Z_fft = np.fft.fft2(received_waveform)
    max_idx = np.unravel_index(np.argmax(np.abs(Z_fft)), Z_fft.shape)
    init_range_idx, init_doppler_idx = max_idx

    current_waveform = received_waveform.copy()

    for k in range(num_of_targets):
        # Initial parameter estimates
        init_range_delay = (init_range_idx) / fs
        init_doppler_freq = (init_doppler_idx - N_sym / 2) / (N_sym * T_prbs)
        init_amplitude = np.abs(current_waveform[init_range_idx, init_doppler_idx])

        # Define likelihood function
        def mle_likelihood(params):
            range_delay, doppler_freq, amplitude = params

            # Generate expected signal
            expected_signal = amplitude * np.exp(
                1j
                * 2
                * np.pi
                * (
                    range_delay * fs * t_fast[:, np.newaxis]
                    + doppler_freq * t_slow[np.newaxis, :]
                )
            )

            # Negative log-likelihood
            residual = current_waveform - expected_signal
            return np.sum(np.abs(residual) ** 2) / noise_power

        # Optimize
        initial_params = [init_range_delay, init_doppler_freq, init_amplitude]

        try:
            result = optimize.minimize(
                mle_likelihood,
                initial_params,
                method="L-BFGS-B",
                options={"disp": False},
            )
            optimal_params = result.x

            # Extract results
            hat_r_tars[k] = optimal_params[0] * speed_of_light / 2
            hat_v_tars[k] = dop2speed(optimal_params[1] / 4, wavelength)

            # Remove detected signal for next iteration
            if k < num_of_targets - 1:
                estimated_signal = optimal_params[2] * np.exp(
                    1j
                    * 2
                    * np.pi
                    * (
                        optimal_params[0] * fs * t_fast[:, np.newaxis]
                        + optimal_params[1] * t_slow[np.newaxis, :]
                    )
                )
                current_waveform = current_waveform - estimated_signal

        except Exception as e:
            warnings.warn(f"MLE optimization failed for target {k}: {e}")
            break

    return hat_r_tars, hat_v_tars


# 5. MAP Method
def estimate_range_doppler_map(
    received_waveform, noise_power, pfa, T_prbs, num_of_targets, fs, wavelength
):
    """MAP Range-Doppler estimation"""

    N_chip, N_sym = received_waveform.shape
    hat_r_tars = np.zeros(num_of_targets)
    hat_v_tars = np.zeros(num_of_targets)

    # Create range bins
    fasttime = np.arange(0, T_prbs, 1 / fs)
    rangebins = speed_of_light * fasttime / 2

    # Time grids
    t_fast = np.arange(N_chip) / fs
    t_slow = np.arange(N_sym) * T_prbs

    # Prior parameters (adjust based on scenario)
    max_range = np.max(rangebins)
    prior_range_mean = max_range / 2
    prior_range_var = (max_range / 4) ** 2
    prior_doppler_mean = 0
    prior_doppler_var = (1 / (4 * T_prbs)) ** 2

    # Initial estimates using 2D FFT
    Z_fft = np.fft.fft2(received_waveform)
    max_idx = np.unravel_index(np.argmax(np.abs(Z_fft)), Z_fft.shape)
    init_range_idx, init_doppler_idx = max_idx

    current_waveform = received_waveform.copy()

    for k in range(num_of_targets):
        # Initial parameter estimates
        init_range_delay = (init_range_idx) / fs
        init_doppler_freq = (init_doppler_idx - N_sym / 2) / (N_sym * T_prbs)
        init_amplitude = np.abs(current_waveform[init_range_idx, init_doppler_idx])

        # Define posterior function
        def map_posterior(params):
            range_delay, doppler_freq, amplitude = params

            # Likelihood term
            expected_signal = amplitude * np.exp(
                1j
                * 2
                * np.pi
                * (
                    range_delay * fs * t_fast[:, np.newaxis]
                    + doppler_freq * t_slow[np.newaxis, :]
                )
            )
            residual = current_waveform - expected_signal
            likelihood_term = np.sum(np.abs(residual) ** 2) / noise_power

            # Prior terms
            range_prior = (range_delay - prior_range_mean) ** 2 / (2 * prior_range_var)
            doppler_prior = (doppler_freq - prior_doppler_mean) ** 2 / (
                2 * prior_doppler_var
            )

            # Negative log posterior
            return likelihood_term + range_prior + doppler_prior

        # Optimize
        initial_params = [init_range_delay, init_doppler_freq, init_amplitude]

        try:
            result = optimize.minimize(
                map_posterior,
                initial_params,
                method="L-BFGS-B",
                options={"disp": False},
            )
            optimal_params = result.x

            # Extract results
            hat_r_tars[k] = optimal_params[0] * speed_of_light / 2
            hat_v_tars[k] = dop2speed(optimal_params[1] / 4, wavelength)

            # Remove detected signal for next iteration
            if k < num_of_targets - 1:
                estimated_signal = optimal_params[2] * np.exp(
                    1j
                    * 2
                    * np.pi
                    * (
                        optimal_params[0] * fs * t_fast[:, np.newaxis]
                        + optimal_params[1] * t_slow[np.newaxis, :]
                    )
                )
                current_waveform = current_waveform - estimated_signal

        except Exception as e:
            warnings.warn(f"MAP optimization failed for target {k}: {e}")
            break

    return hat_r_tars, hat_v_tars


def pulsint(data, integration_type="noncoherent"):
    """Pulse integration - simplified implementation"""
    if integration_type == "noncoherent":
        return np.abs(data) ** 2
    else:
        return data


def estimate_range_doppler_original(
    received_waveform, noise_power, pfa, T_prbs, num_of_targets, fs, wavelength
):
    """Original method converted from MATLAB"""

    hat_r_tars = np.zeros(num_of_targets)
    hat_v_tars = np.zeros(num_of_targets)

    # Get dimensions
    N_chip, N_sym = received_waveform.shape

    # Create time grid and range bins
    fasttime = np.arange(0, T_prbs, 1 / fs)
    rangebins = speed_of_light * fasttime / 2

    # Calculate threshold
    thresh_db = npwgnthresh(pfa, N_sym, "noncoherent")
    thresh = noise_power * db2pow(thresh_db)

    # Find peak with Sum - sum over first 32 symbols (or all if less than 32)
    n_sum = min(32, N_sym)
    sum_received_waveform = np.sum(np.abs(received_waveform[:, :n_sum]), axis=1)

    # Pulse integration (noncoherent)
    integrated_signal = pulsint(sum_received_waveform, "noncoherent")

    # Find peaks using scipy
    peaks, properties = signal.find_peaks(integrated_signal, height=thresh)

    if len(peaks) > 0:
        # Sort peaks by height (descending)
        peak_heights = integrated_signal[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # Descending order
        range_detect = peaks[sorted_indices]

        # Get best peaks (up to num_of_targets)
        try:
            best_peak = range_detect[:num_of_targets]
            if len(best_peak) < num_of_targets:
                # Pad with zeros if not enough peaks found
                temp = np.zeros(num_of_targets, dtype=int)
                temp[: len(best_peak)] = best_peak
                best_peak = temp
        except:
            best_peak = np.zeros(num_of_targets, dtype=int)
    else:
        best_peak = np.zeros(num_of_targets, dtype=int)

    # Process each target
    for k in range(num_of_targets):
        if best_peak[k] > 0:
            idx = best_peak[k]

            # Extract slow-time signal
            ts = received_waveform[idx, :].squeeze()  # Slow-time signal

            # Compute periodogram using scipy.signal.periodogram
            # MATLAB: [Pxx, F] = periodogram(ts, [], N_sym, 1/T_prbs, 'centered')
            f, Pxx = signal.periodogram(
                ts, fs=1 / T_prbs, nfft=N_sym, return_onesided=False, scaling="density"
            )

            # Center the frequencies (equivalent to 'centered' in MATLAB)
            f = np.fft.fftshift(f)
            Pxx = np.fft.fftshift(Pxx)

            # Find maximum
            max_idx = np.argmax(Pxx)
            doppler_freq = f[max_idx]  # Hz

            # Convert to velocity
            hat_v_tars[k] = dop2speed(doppler_freq / 4, wavelength)  # m/s

            # Get range
            if idx < len(rangebins):
                hat_r_tars[k] = rangebins[idx]  # meters
            else:
                hat_r_tars[k] = 0
        else:
            hat_r_tars[k] = 0  # Default range
            hat_v_tars[k] = 0  # Default velocity

    return hat_r_tars, hat_v_tars


# Example usage
if __name__ == "__main__":
    # Example parameters
    N_chip, N_sym = 256, 64
    noise_power = 1.0
    pfa = 1e-6
    T_prbs = 1e-3
    num_of_targets = 2
    fs = 1e6
    wavelength = 0.1

    # Create example received waveform
    received_waveform = np.random.randn(N_chip, N_sym) + 1j * np.random.randn(
        N_chip, N_sym
    )

    # Test original method
    print("Testing original method...")
    ranges, velocities = estimate_range_doppler_original(
        received_waveform, noise_power, pfa, T_prbs, num_of_targets, fs, wavelength
    )
    print(f"Original: Ranges = {ranges}, Velocities = {velocities}")

    # Test all methods
    print("\nTesting all methods...")

    methods = [
        ("Original", estimate_range_doppler_original),
        ("2D FFT", estimate_range_doppler_2dfft),
        ("2D MUSIC", estimate_range_doppler_2dmusic),
        ("CLEAN", estimate_range_doppler_clean),
        ("MLE", estimate_range_doppler_mle),
        ("MAP", estimate_range_doppler_map),
    ]

    for name, method in methods:
        try:
            ranges, velocities = method(
                received_waveform,
                noise_power,
                pfa,
                T_prbs,
                num_of_targets,
                fs,
                wavelength,
            )
            print(f"{name}: Ranges = {ranges}, Velocities = {velocities}")
        except Exception as e:
            print(f"{name}: Failed with error {e}")
