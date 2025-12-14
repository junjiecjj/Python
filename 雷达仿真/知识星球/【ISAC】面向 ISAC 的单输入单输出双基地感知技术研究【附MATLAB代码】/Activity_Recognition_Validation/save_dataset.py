import os
import numpy as np
import time
from scipy.interpolate import interp1d
from scipy.signal import zoom_fft
import csiread

# ===== Global Params =====
c = 299792458
frequency_spacing = 312.5e3
Fs = 1000.0

subcarrier_index_L = [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1]
subcarrier_index_H = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28]
subcarrier_index = np.array(subcarrier_index_L + subcarrier_index_H)
target_index = np.arange(-28, 29, 1)
freq_sub = target_index * frequency_spacing

win, inc = 128, 32
Range_BINS_OFFSET = 64
sigma = 32

f1_range, f2_range, Range_BINS = 0, 16, 32
rangeArray = np.linspace(f1_range, f2_range, Range_BINS)
range_vector = np.exp(-1j * 2 * np.pi * freq_sub.reshape(-1, 1) / c * rangeArray.reshape(1, -1))

Doppler_BINS = win
f1_doppler, f2_doppler = -150, 150
dopplerArray = np.linspace(f1_doppler, f2_doppler, Doppler_BINS)

csi_len = 128


def process_csi_to_microdoppler(csi_trans):
    """
    csi_trans : (N, 90) ndarray, each row reshaped to (30, 3)
    return: micro_doppler (M, 128, 3), est_range (3,)
    """
    csi_list = []
    doppler_range_heatmap_list = []

    for i in range(csi_trans.shape[0]):
        csi = csi_trans[i, :].reshape(3, 30).T

        interp_func = interp1d(subcarrier_index, csi, kind='linear', axis=0, fill_value="extrapolate")
        csi_interpolated = interp_func(target_index)

        range_ifft = np.fft.ifft(csi_interpolated, n=Range_BINS_OFFSET, axis=0)
        power_delay = np.abs(range_ifft) ** 2

        max_bin_idx = np.argmax(power_delay, axis=0)
        x = np.arange(Range_BINS_OFFSET)
        gaussian_window = np.exp(-0.5 * ((x[:, None] - max_bin_idx[None, :]) / sigma) ** 2)
        range_filtered = range_ifft * gaussian_window

        csi_filtered = np.fft.fft(range_filtered, n=Range_BINS_OFFSET, axis=0)
        csi_filtered = csi_filtered[:len(target_index), :]

        CSI = csi_interpolated * np.conj(csi_filtered)
        csi_list.append(CSI)

        if len(csi_list) >= win:
            csi_array = np.array(csi_list)
            csi_array_avg = np.mean(csi_array, axis=0, keepdims=True)
            csi_array = (csi_array - csi_array_avg) / csi_array_avg
            csi_list = csi_list[inc:]

            doppler_range_heatmap = np.zeros((Doppler_BINS, Range_BINS, 3), dtype=float)
            for ant in range(3):
                cln_csi_array = csi_array[:, :, ant].T
                cln_csi_array = np.concatenate((cln_csi_array, np.conj(cln_csi_array)), axis=1)

                Rxx = cln_csi_array @ np.conjugate(cln_csi_array.T)
                reg_param = np.trace(Rxx) / Rxx.shape[0] * 0.5
                J = np.fliplr(np.eye(57))
                Rxx = 0.5 * (Rxx + J @ np.conjugate(Rxx) @ J) + reg_param * np.eye(Rxx.shape[1])

                Rxx_inv = np.linalg.inv(Rxx)
                numerator = Rxx_inv @ range_vector
                denominator = np.einsum('ij,ij->j', range_vector.conj(), numerator)
                weights = numerator / (denominator + 1e-12)

                weighted_csi = (weights.conj().T @ cln_csi_array).T
                weighted_csi = 0.5 * (weighted_csi[0:Doppler_BINS, :] + weighted_csi[Doppler_BINS:, :])

                doppler_fft = zoom_fft(weighted_csi, [f1_doppler, f2_doppler], Doppler_BINS, fs=Fs, axis=0)
                doppler_range_heatmap[:, :, ant] = np.abs(doppler_fft)

            doppler_range_heatmap_list.append(doppler_range_heatmap)

    doppler_range_heatmap_array = np.array(doppler_range_heatmap_list)
    micro_doppler = np.mean(doppler_range_heatmap_array, axis=2)
    est_range = rangeArray[np.argmax(np.max(doppler_range_heatmap_array, axis=(0, 1)), axis=0)]
    return micro_doppler, est_range

# ======================== #
# ----- File Walker & Save ----- #
# ======================== #
def collect_and_write(root_dir):
    file_count = 0
    found_dat = 0
    folder_name = os.path.basename(root_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.dat'):
                found_dat += 1
                full_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(full_path, root_dir)
                print(f"[Found] {rel_path}")

                try:
                    csidata = csiread.Intel(full_path, nrxnum=3, ntxnum=1)
                    csidata.read()
                    csi = csidata.get_scaled_csi()[:, :, :, 0]
                    csi_trans = np.reshape(csi.transpose(0, 2, 1), (csi.shape[0], 90))

                    if csi_trans.shape[0] < csi_len:
                        print(f"[Skip] Too short: {rel_path}")
                        continue

                    print("-----------------------------------------------------------")
                    start_time = time.time()
                    micro_doppler, est_range = process_csi_to_microdoppler(csi_trans)
                    end_time = time.time()
                    print(f"[Info] Execution Time: {end_time - start_time:.4f} seconds")

                    rel_path_npy = os.path.splitext(rel_path)[0] + '.npy'
                    save_path = os.path.join(folder_name, rel_path_npy)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, {'micro_doppler': micro_doppler, 'range': est_range})
                    print(f"[OK] Saved: {save_path}")
                    file_count += 1
                except Exception as e:
                    print(f"[Error] {rel_path} → {e}")

    print(f"\n🔍 Found {found_dat} .dat files, ✅ Saved {file_count} features.\n")


# ======================== #
# --------- Main ---------- #
# ======================== #
if __name__ == '__main__':
    # List of target data directories
    root_dir_list = [
        '20181109', '20181112',
        '20181115', '20181116', '20181117', '20181118',
        '20181121', '20181127', '20181128',
        '20181130', '20181204', '20181205_2',
        '20181205_3', '20181208', '20181209', '20181211'
    ]

    base_path = '..'  # Parent directory containing the above folders

    for root_dir in root_dir_list:
        full_root_dir = os.path.join(base_path, root_dir)
        print(f"\n=== Processing {full_root_dir} ===")
        collect_and_write(full_root_dir)
