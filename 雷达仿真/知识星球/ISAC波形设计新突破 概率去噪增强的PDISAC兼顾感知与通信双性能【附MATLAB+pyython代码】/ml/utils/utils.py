import torch
import numpy as np
import scipy.constants as const
from geopy.distance import geodesic


def distance_3d(pos1, pos2):
    # Horizontal ground distance (ignores altitude)
    ground_dist = geodesic((pos1[0], pos1[1]), (pos2[0], pos2[1])).meters
    # Altitude difference
    alt_diff = pos1[2] - pos2[2]
    # 3D distance
    return np.sqrt(ground_dist**2 + alt_diff**2)


def compute_nmse(pred, target, dim=1):
    mse = torch.mean((pred - target) ** 2, dim=dim)
    target_power = torch.mean(target**2, dim=dim)
    nmse = mse / (target_power + 1e-8)  # Add small epsilon to avoid division by zero
    return nmse.mean()


def preprocess(x, domain="freq", dims=[0, 1]):
    if domain == "time":
        # FFT along slow-time axis (dim=1), shift
        rd_map = torch.fft.fftshift(torch.fft.fft(x, dim=dims[1]), dim=dims[1])
        return rd_map
    elif domain == "freq":
        # Doppler FFT along slow-time axis (dim=1)
        doppler_freq = torch.fft.fftshift(torch.fft.fft(x, dim=dims[1]), dim=dims[1])
        # IFFT along fast-time axis (dim=0)
        rd_map = torch.fft.ifft(doppler_freq, dim=dims[0])
        return rd_map
    else:
        raise ValueError(f"Unknown domain: {domain}")


def postprocess(rd_map, domain="freq", dims=[0, 1]):
    if domain == "time":
        # Inverse of FFT (dim=1) with shift
        time_data = torch.fft.ifft(
            torch.fft.ifftshift(rd_map, dim=dims[1]), dim=dims[1]
        )
        return time_data
    elif domain == "freq":
        # Undo Doppler processing
        doppler_time = torch.fft.fft(rd_map, dim=dims[0])  # FFT along fast-time
        time_data = torch.fft.ifft(
            torch.fft.ifftshift(doppler_time, dim=dims[1]), dim=dims[1]
        )
        return time_data
    else:
        raise ValueError(f"Unknown domain: {domain}")


def convert_to_complex(x, dim=0):
    if dim == 0:
        return torch.complex(x[0, ...], x[1, ...])
    elif dim == 1:
        return torch.complex(x[:, 0, ...], x[:, 1, ...])
    else:
        raise ValueError(f"Unknown dim {dim} of {x.shape}")


def reverse_from_complex(x, dim=0):
    real_part = x.real
    imag_part = x.imag
    if dim == 0:
        return torch.stack([real_part, imag_part], dim=0)
    elif dim == 1:
        return torch.stack([real_part, imag_part], dim=1)
    else:
        raise ValueError(f"Unknown dim {dim} of {x.shape}")


def time_to_freq(x, dim=1):
    if isinstance(x, torch.Tensor):
        X = torch.fft.fft(x, dim=dim)
    elif isinstance(x, np.ndarray):
        X = np.fft.fft(x, axis=dim)
    return X


def freq_to_time(X, dim=1):
    if isinstance(X, torch.Tensor):
        x = torch.fft.ifft(X, dim=dim)
    elif isinstance(X, np.ndarray):
        x = np.fft.ifft(X, axis=dim)
    return x


def complex_mul_conj(Y, P, domain="freq"):
    if domain == "freq":
        Y = convert_to_complex(Y, dim=1)
        P = convert_to_complex(P, dim=1)

        Z = Y * torch.conj(P[:, :, :1])
        return reverse_from_complex(Z, dim=1)
    elif domain == "time":
        Y = time_to_freq(convert_to_complex(Y, dim=1), dim=1)
        P = time_to_freq(convert_to_complex(P, dim=1), dim=1)
        Z = Y * torch.conj(P[:, :, :1])
        Z = freq_to_time(Z, dim=1)
        return reverse_from_complex(Z, dim=1)
    else:
        raise ValueError(f"Unknown domain: {domain}")
