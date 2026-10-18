import torch
from torch.utils.data import Dataset
import numpy as np
import random
from scipy.ndimage import shift, zoom, gaussian_filter, map_coordinates


class Gesture_Dataset(Dataset):
    def __init__(self, df_data, csi_len=64, doppler_len=128, augment=True):
        """
        Custom Dataset for micro-Doppler gesture classification.

        Args:
            df_data (DataFrame): Metadata containing file paths and gesture labels.
            csi_len (int): Target time dimension length after resizing.
            doppler_len (int): Target Doppler dimension length after resizing.
            augment (bool): Whether to apply data augmentation.
        """
        self.df = df_data.reset_index(drop=True)
        self.csi_len = csi_len
        self.doppler_len = doppler_len
        self.augment = augment

        # Map gesture names to integer labels (starting from 0)
        self.gesture_classes = sorted(self.df['gesture_name'].unique().tolist())
        self.gesture_to_label = {g: i for i, g in enumerate(self.gesture_classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.iloc[idx]["FilePath"]
        gesture = self.df.iloc[idx]["gesture_name"]
        label_val = self.gesture_to_label[gesture]

        # Load micro-Doppler tensor from .npy file
        data = np.load(file_path, allow_pickle=True).item()
        micro_doppler_all = data['micro_doppler']

        # Skip invalid samples (wrong shape or too short)
        if (not isinstance(micro_doppler_all, np.ndarray) or
                micro_doppler_all.ndim != 3 or
                micro_doppler_all.shape[0] < 16):
            return self.__getitem__((idx + 1) % len(self))

        # Randomly select one view/channel from the 3 available
        micro_doppler = micro_doppler_all[:, :, random.randint(0, 2)]

        # Apply random augmentation if enabled
        aug_type = random.choice([
            "None", "TimeReverse", "DopplerReverse",
            "TimeShift_Left", "TimeShift_Right",
            "AddNoise", "TimeStretch", "DopplerStretch", "ElasticDistortion"
        ]) if self.augment else "None"
        micro_doppler = self.apply_augmentation(micro_doppler, aug_type)

        # Resize to target (csi_len, doppler_len)
        micro_doppler = self.resize_microdoppler(micro_doppler)

        # Replace NaN or Inf values with 0
        micro_doppler = np.nan_to_num(micro_doppler, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize (mean=0, std=1)
        mean_val = np.mean(micro_doppler)
        std_val = np.std(micro_doppler)
        if not np.isfinite(mean_val) or not np.isfinite(std_val) or std_val < 1e-6:
            micro_doppler = np.zeros_like(micro_doppler)
        else:
            micro_doppler = (micro_doppler - mean_val) / std_val

        # Add channel dimension → shape: (1, H, W)
        micro_doppler_tensor = torch.tensor(micro_doppler, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_val, dtype=torch.long)
        return micro_doppler_tensor, label_tensor

    def resize_microdoppler(self, x):
        """Resize the input micro-Doppler spectrogram to target size using bilinear interpolation."""
        target_T = self.csi_len
        target_F = self.doppler_len
        T, F = x.shape

        if T == 0 or F == 0:
            return np.zeros((target_T, target_F))

        zoom_factors = (target_T / T, target_F / F)
        return zoom(x, zoom=zoom_factors, order=1)

    def elastic_distortion(self, x, alpha=15, sigma=3):
        """Apply elastic distortion to the input spectrogram."""
        T, F = x.shape
        dx = gaussian_filter((np.random.rand(T, F) * 2 - 1), sigma=sigma, mode="constant") * alpha
        dy = gaussian_filter((np.random.rand(T, F) * 2 - 1), sigma=sigma, mode="constant") * alpha
        x_grid, y_grid = np.meshgrid(np.arange(F), np.arange(T))
        indices = np.reshape(y_grid + dy, (-1, 1)), np.reshape(x_grid + dx, (-1, 1))
        return map_coordinates(x, indices, order=1, mode='reflect').reshape((T, F))

    def apply_augmentation(self, x, mode):
        """Apply a specific augmentation mode to the input spectrogram."""
        if mode == "None":
            return x
        elif mode == "TimeReverse":
            return x[::-1, :]
        elif mode == "DopplerReverse":
            return x[:, ::-1]
        elif mode == "TimeShift_Left":
            shift_amount = -random.randint(x.shape[0] // 10, x.shape[0] // 3)
            return shift(x, shift=(shift_amount, 0), mode='nearest')
        elif mode == "TimeShift_Right":
            shift_amount = random.randint(x.shape[0] // 10, x.shape[0] // 3)
            return shift(x, shift=(shift_amount, 0), mode='nearest')
        elif mode == "AddNoise":
            snr_db = random.choice([10, 20, 30])
            signal_power = np.mean(x ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
            return x + noise
        elif mode == "TimeStretch":
            scale = random.uniform(0.8, 1.2)
            return zoom(x, zoom=(scale, 1), order=1)
        elif mode == "DopplerStretch":
            scale = random.uniform(0.8, 1.2)
            return zoom(x, zoom=(1, scale), order=1)
        elif mode == "ElasticDistortion":
            return self.elastic_distortion(x)
        else:
            return x
