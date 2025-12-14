# ===============================
# This code is based on "Towards SISO Bistatic Sensing for ISAC"
# and implements Delay, Doppler, and Micro-Doppler feature extraction.

# ===============================
import csiread  # Intel CSI reader
import numpy as np
import time
from collections import deque
from scipy.interpolate import interp1d
from scipy.signal import zoom_fft
import matplotlib.pyplot as plt

# ---------- 1) SRCC: main-path correction ----------
def srcc_correction(csi):
    """
    SRCC main path correction to remove TO/CFO.
    Input:
        csi : array-like, shape (30,) or (30,1)  (802.11n 20MHz usable subcarriers)
    Return:
        1-D complex array, length 57 (interpolated to -28..+28)
    """
    subcarrier_index_L = [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1]
    subcarrier_index_H = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28]
    subcarrier_index = np.array(subcarrier_index_L + subcarrier_index_H)
    target_index = np.arange(-28, 29, 1)

    ifft_win = 128
    sigma = 64.0
    interp_kind = "linear"

    csi = np.asarray(csi)
    if csi.ndim == 1:
        csi = csi[:, None]
    elif not (csi.ndim == 2 and csi.shape[1] == 1):
        raise ValueError("csi must have shape (N_subc,) or (N_subc, 1)")
    if csi.shape[0] != len(subcarrier_index):
        raise ValueError(f"csi rows must equal len(subcarrier_index) = {len(subcarrier_index)}")

    # interpolate onto target subcarriers
    f = interp1d(subcarrier_index, csi, kind=interp_kind, axis=0,
                 fill_value="extrapolate", assume_sorted=False)
    csi_interp = f(target_index)  # (57,1)

    # find main path in delay domain and window it
    x_ifft = np.fft.ifft(csi_interp, n=ifft_win, axis=0)  # (ifft_win,1)
    pdp = np.abs(x_ifft) ** 2
    peak = np.argmax(pdp, axis=0)
    x = np.arange(ifft_win)[:, None]
    w = np.exp(-0.5 * ((x - peak[None, :]) / sigma) ** 2)

    x_win = x_ifft * w
    csi_filt = np.fft.fft(x_win, n=ifft_win, axis=0)[:len(target_index), :]

    CSI = csi_interp * np.conj(csi_filt)  # phase aligned to main path
    return CSI[:, 0]  # (57,)


# ---------- 2) Range-Doppler MVDR (all params inside) ----------
def range_doppler_mvdr(csi_array):
    """
    Compute Range-Doppler heatmap with MVDR and Doppler zoom-FFT.
    All WiFi/system parameters are fixed inside.

    Input:
        csi_array : np.ndarray, shape (N_snapshots, N_subcarriers=57 or 30)
                    If 30, please pre-call srcc_correction per snapshot & stack.
    Return:
        np.ndarray, shape (Doppler_BINS=128, Range_BINS=32)
    """
    # WiFi/system params
    c = 299792458.0  # m/s
    freq_center = 5.32e9  # Hz
    frequency_spacing = 312_500.0  # Hz
    Fs = 1000.0  # Hz
    win = 128  # Doppler_BINS

    # target subcarriers & their frequencies
    target_index = np.arange(-28, 29, 1)
    freq_sub_interpolated = freq_center + target_index * frequency_spacing  # (57,)

    # Range steering vector
    Range_BINS = 32
    f1_range, f2_range = 0.0, 32.0
    rangeArray = np.linspace(f1_range, f2_range, Range_BINS)
    range_vector = np.exp(-1j * 2 * np.pi * freq_sub_interpolated.reshape(-1, 1) / c
                          * rangeArray.reshape(1, -1))  # (57,32)

    # Doppler params
    Doppler_BINS = win
    f1_doppler, f2_doppler = -150.0, 150.0

    # Input normalization / shape
    X = np.asarray(csi_array)
    if X.ndim != 2:
        raise ValueError("csi_array must be 2-D: (N_snapshots, N_subcarriers)")
    if X.shape[1] not in (57, 30):
        raise ValueError("N_subcarriers must be 57 (interpolated) or 30 (raw).")
    if X.shape[1] == 30:
        raise ValueError("Please convert 30-subcarrier CSI to 57 with srcc_correction per snapshot.")

    # 1) static removal
    m = np.mean(X, axis=0)
    X = (X - m) / m

    # 2) covariance with forward-backward averaging
    C = X.T  # (57, N)
    C = np.concatenate((C, np.conj(C)), 1)  # (57, 2N)
    Rxx = C @ np.conj(C.T)  # (57,57)
    reg = np.trace(Rxx) / Rxx.shape[0] * 0.5
    J = np.fliplr(np.eye(Rxx.shape[0]))
    Rxx = 0.5 * (Rxx + J @ np.conj(Rxx) @ J) + reg * np.eye(Rxx.shape[0])

    # 3) MVDR weights
    Rinv = np.linalg.inv(Rxx)
    num = Rinv @ range_vector
    den = np.einsum("ij,ij->j", np.conj(range_vector), num)
    W = num / (den + 1e-12)  # (57,32)

    # 4) apply weights, fold conjugate snapshots
    Y = (np.conj(W).T @ C).T  # (2N,32)
    # assume first win and next win are conjugate pairs as in your pipeline
    if Y.shape[0] < 256:
        raise ValueError("Require at least 256 snapshots for folding (128 + 128).")
    Y = Y[0:128, :] + Y[128:256, :]  # (128,32)

    # 5) Doppler zoom-FFT
    Z = zoom_fft(Y, [f1_doppler, f2_doppler], Doppler_BINS, fs=Fs, axis=0)
    return np.abs(Z)  # (128,32)

def main():
    # ---------------- Fixed constants (DO NOT CHANGE) ----------------
    C_LIGHT = 299_792_458  # Speed of light (m/s)
    FREQ_CENTER = 5.32e9  # WiFi center frequency (Hz)
    Fs = 1000  # Sampling rate in Hz
    WIN = 128  # CPI window length (frames per window)
    INC = 128  # Hop size (non-overlapping windows)

    RANGE_BINS = 32  # Number of range bins
    F1_RANGE = 0  # Range start (m)
    F2_RANGE = 32  # Range end (m)

    DOPPLER_BINS = WIN  # Doppler bins (tied to window length)
    F1_DOPPLER = -150  # Doppler start (Hz)
    F2_DOPPLER = 150  # Doppler end (Hz)
    # ----------------------------------------------------------------

    # ---------------- Build range & Doppler axes for plotting --------
    # Range axis in meters
    # Note that the range is calculated as the distance from the human target to the Tx–Rx pair,
    # minus the distance between the transmitter and receiver.
    range_axis = np.linspace(F1_RANGE, F2_RANGE, RANGE_BINS)

    # Doppler axis in Hz, later mapped to m/s via v = f_d * c / f_c
    doppler_bins = np.linspace(F1_DOPPLER, F2_DOPPLER, DOPPLER_BINS)
    doppler_axis_ms = doppler_bins * C_LIGHT / FREQ_CENTER

    print(f"[info] Range axis (bins): [{F1_RANGE}, {F2_RANGE}] -> {RANGE_BINS} bins (fixed)")
    print(f"[info] Doppler axis (bins): [{F1_DOPPLER}, {F2_DOPPLER}] -> {DOPPLER_BINS} bins (fixed)")
    print(f"[info] Doppler axis mapped to velocity: "
          f"min={doppler_axis_ms.min():.3f} m/s, max={doppler_axis_ms.max():.3f} m/s (fixed)")

    # ---------------- Load CSI dataset ----------------
    # The provided dataset (Raw CSI data) is based on a 1Tx–3Rx antenna configuration.
    # The target's motion trajectory follows an elliptical path, as described in:
    '''
    @article{wang2022single,
      title={Single-target real-time passive WiFi tracking},
      author={Wang, Zhongqin and Zhang, J Andrew and Xu, Min and Guo, Y Jay},
      journal={IEEE Transactions on Mobile Computing},
      year={2023},
      volume={22},
      number={6},
      pages={3724-3742},
      publisher={IEEE}
    }
    
    @article{wang2024passive,
      title={Passive Human Tracking With WiFi Point Clouds},
      author={Wang, Zhongqin and Zhang, J. Andrew and Zhang, Haimin and Xu, Min and Guo, Jay},
      journal={IEEE Internet of Things Journal}, 
      year={2025},
      volume={12},
      number={5},
      pages={5528-5543}
    }
    '''
    dat_file = "./ellipse_1.dat"
    # Parse Intel 5300 CSI; choose TX=0 (shape -> packets x subcarriers x Rx x Tx)
    csidata = csiread.Intel(dat_file, nrxnum=3, ntxnum=1)
    csidata.read()
    csi = csidata.get_scaled_csi()  # (packets, subc, rx, tx)
    csi_ = csi[:, :, :, 0]  # (packets, subcarriers, Rx)
    print(f"[info] Packets: {csi_.shape[0]}, Subcarriers: {csi_.shape[1]}, RX antennas: {csi_.shape[2]}")

    # --------------- User configuration ---------------
    # Our WiDFS3.0 is designed for 1Tx-1Rx single-antenna processing;
    # We pick one RX antenna (0, 1, or 2).
    I_ANT = 2
    # ---------------------------------------------------

    # ---------------- Sliding window buffers ----------------
    window_buf = deque(maxlen=WIN)  # Collect WIN corrected frames
    ranges_list = []  # Range peak per processed window
    dpl_list = []  # Doppler (velocity) peak per processed window
    micro_dpl_list = []  # Per-window micro-Doppler (mean over range)
    processed_windows = 0  # Counter of processed windows

    # ---------------- Main processing loop ----------------
    # Iterate over packets; starts at index 10000.
    for j_ofdm in range(10000, csi_.shape[0]):
        # 1) Take one packet on selected RX; shape: (subcarriers,)
        csi_each = csi_[j_ofdm, :, I_ANT]

        # 2) SRCC for random phase offset removal
        csi_cc = srcc_correction(csi_each)

        # 3) Push into sliding window
        window_buf.append(csi_cc)

        # 4) Process once we have a full window (WIN frames)
        if len(window_buf) == WIN:

            # Stack to shape (WIN, 57)
            csi_array = np.vstack(window_buf)

            # 5) Range–Doppler feature
            t0 = time.time()  # Start timing for this iteration
            rd_map = range_doppler_mvdr(csi_array)  # expected shape: (128, 32)
            # Print elapsed time for this window
            print(f"[time] Elapsed: {time.time() - t0:.5f} s")

            # 6) Record
            # Extract a point with the maximum power from the Range–Doppler feature map
            peak_idx = np.unravel_index(np.argmax(rd_map), rd_map.shape)  # (doppler_idx, range_idx)
            peak_doppler_ms = float(doppler_axis_ms[peak_idx[0]])  # m/s
            peak_range_bin = float(range_axis[peak_idx[1]])  # m

            dpl_list.append(peak_doppler_ms)
            ranges_list.append(peak_range_bin)
            micro_dpl_list.append(np.mean(rd_map, axis=1))  # (DOPPLER_BINS,)

            processed_windows += 1
            print(f"[prog] Processed windows: {processed_windows}, "
                  f"range={peak_range_bin:.2f} m, doppler={peak_doppler_ms:.3f} m/s")

            # 7) Slide forward by INC frames (non-overlapping)
            for _ in range(min(INC, len(window_buf))):
                window_buf.popleft()

    # ---------------- Post-processing checks ----------------
    # Convert lists to arrays for plotting
    ranges_array = np.asarray(ranges_list, dtype=float)  # shape: (N_win,)
    dpl_array = np.asarray(dpl_list, dtype=float)  # shape: (N_win,)
    micro_dpl_array = np.asarray(micro_dpl_list)  # shape: (N_win, DOPPLER_BINS)

    # ---------------- Plotting ----------------
    # Time axis per window: each window represents WIN/Fs seconds of data
    t_axis = np.arange(ranges_array.shape[0]) * (WIN / Fs)

    # 1) Range trajectory over time
    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, ranges_array)
    plt.title("Estimated Range")
    plt.xlabel("Time (s)")
    plt.ylabel("Range (m)")
    plt.grid(True)
    plt.tight_layout()

    # 2) Doppler velocity over time
    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, dpl_array, color="orange")
    plt.title("Estimated Doppler Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.tight_layout()

    # 3) Micro-Doppler heatmap (mean over range per window)
    plt.figure(figsize=(10, 5))
    md = micro_dpl_array.T  # (DOPPLER_BINS, N_win)
    md = np.log1p(md / np.max(md))  # log compression for better visual dynamics
    ax = plt.contourf(t_axis, doppler_axis_ms, md, cmap='jet',
                      levels=np.linspace(md.min(), md.max(), 50))
    plt.colorbar(ax, label="log1p(normalized power)")

    plt.xlabel('Time (s)')
    plt.ylabel('Doppler (m/s)')
    plt.title('Micro-Doppler Heatmap')
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
