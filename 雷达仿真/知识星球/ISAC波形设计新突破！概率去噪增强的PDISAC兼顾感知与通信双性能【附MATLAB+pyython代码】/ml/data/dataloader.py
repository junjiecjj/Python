from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from utils.resize import ComplexMatrixResizer
from utils.scales import MinMaxScaler, scale_complex_matrices
from configs.isac import p_freq_path, matrix_size
from configs.ml import domain
from utils.utils import freq_to_time


class PRBSDataset(Dataset):
    def __init__(
        self,
        root_dir,
        categories,
        snrs=None,
        transform_funcs=None,
        p_freq_path=p_freq_path,
        size=matrix_size,
    ):
        self.root_dir = Path(root_dir)
        self.categories = categories
        self.ignored_categories = [
            "P_prbs",
            "Z_PRBS_waveform_no_noise",
            "Z_PRBS_waveform",
        ]
        self.p_freq_path = p_freq_path
        self.transform_funcs = transform_funcs
        self.snrs = snrs
        self.size = size
        self.samples = []

        # Collect samples for each category, ensuring alignment by filename
        sample_dict = {}
        for category in categories:
            if category in self.ignored_categories:
                continue
            category_dir = self.root_dir / category
            snr_dirs = [d for d in category_dir.rglob("*") if d.is_dir()]

            for snr_dir in snr_dirs:
                # snr = snr_dir.name
                snr = snr_dir.relative_to(category_dir).as_posix()
                if snr in self.snrs or self.snrs is None:
                    for mat_path in snr_dir.glob("*.mat"):
                        if mat_path.is_file():
                            snr = self.sanitize_snr(snr)
                            sample_key = f"{snr}_{mat_path.name}"
                            if sample_key not in sample_dict:
                                sample_dict[sample_key] = {}
                            sample_dict[sample_key][category] = (mat_path, snr)
        # Ensure each sample has data for all categories
        for sample_key, sample_data in sample_dict.items():
            self.samples.append((sample_key, sample_data))
            
    @staticmethod
    def sanitize_snr(snr):
        s = str(snr)
        return s.replace("/", "_").replace("\\", "_")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        _, sample = self.samples[idx]

        datasets = {k: None for k in self.categories}

        # Load P_prbs once
        P = scipy.io.loadmat(self.p_freq_path)["data"]
        datasets["P_prbs"] = P

        # Load Y waveforms
        for k in ["Y_PRBS_waveform", "Y_PRBS_waveform_no_noise"]:
            if k in self.categories:
                mat_path, _ = sample[k]
                Y = scipy.io.loadmat(mat_path)["data"]
                datasets[k] = Y

        # Generate Z via matched filtering
        if "Z_PRBS_waveform" in self.categories:
            Z = datasets["Y_PRBS_waveform"] * np.conj(datasets["P_prbs"])
            datasets["Z_PRBS_waveform"] = Z

        if "Z_PRBS_waveform_no_noise" in self.categories:
            Z = datasets["Y_PRBS_waveform_no_noise"] * np.conj(datasets["P_prbs"])
            datasets["Z_PRBS_waveform_no_noise"] = Z

        # --- Apply transforms ---
        for k in self.categories:
            if domain == "time":
                datasets[k] = freq_to_time(datasets[k], dim=0)
            datasets[k] = self.process_data(datasets[k])

        return tuple(datasets[k] for k in self.categories)

    def process_data(self, data):
        data = scale_complex_matrices(data, 0, 1, scale_magnitude=True)
        for t in self.transform_funcs:
            if isinstance(t, MinMaxScaler):
                data = t.fit_transform(data)
            elif isinstance(t, ComplexMatrixResizer):
                data = t.scale(data, self.size)

        # Complex → (2, H, W)
        if np.iscomplexobj(data):
            data = np.stack([data.real, data.imag], axis=0)

        return torch.tensor(data, dtype=torch.float32)


class PRBSDataloader:
    def __init__(
        self,
        root_dir,
        categories,
        snrs,
        transform_funcs=None,
        batch_size=1,
        pin_memory=True,
        num_worker=1,
        train_test_split=0.8,
        shuffle=True,
        p_freq_path=p_freq_path,
        size=matrix_size,
    ):
        self.dataset = PRBSDataset(
            root_dir=root_dir,
            categories=categories,
            snrs=snrs,
            transform_funcs=transform_funcs,
            p_freq_path=p_freq_path,
            size=size,
        )
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_worker = num_worker
        self.train_test_split = train_test_split
        self.shuffle = shuffle
        self.categories = categories

    def get_dataloaders(self):
        """
        Returns a tuple of (train_loader, test_loader) for the dataset.

        Returns:
            tuple: (train_loader, test_loader) as DataLoader objects.
        """
        indices = list(range(len(self.dataset)))
        train_idx, test_idx = train_test_split(
            indices,
            train_size=self.train_test_split,
            shuffle=self.shuffle,
            random_state=42 if self.shuffle else None,
        )

        train_subset = Subset(self.dataset, train_idx)
        test_subset = Subset(self.dataset, test_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=True,
            num_workers=self.num_worker,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
            num_workers=self.num_worker,
        )

        return train_loader, test_loader

    def get_fold_dataloaders(self):
        """
        Returns a list of (train_loader, val_loader) tuples for each fold.

        Returns:
            list: List of tuples, each containing train and validation DataLoaders.
        """
        fold_loaders = []
        kfold = KFold(
            n_splits=int(1 / (1 - self.train_test_split)), shuffle=self.shuffle
        )

        for train_idx, val_idx in kfold.split(range(len(self.dataset))):
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                shuffle=True,
                num_workers=self.num_worker,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                shuffle=False,
                num_workers=self.num_worker,
            )
            fold_loaders.append((train_loader, val_loader))

        return fold_loaders
