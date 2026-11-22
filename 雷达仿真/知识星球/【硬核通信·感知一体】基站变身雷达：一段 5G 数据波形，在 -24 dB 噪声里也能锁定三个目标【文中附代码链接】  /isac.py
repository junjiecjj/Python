"""OFDM integrated sensing and communication: one waveform does both jobs.

The base station sends normal OFDM symbols, listens to the echoes, and removes
the known symbols to get a range-Doppler map. Comb pilots let us trade sensing
resources against data rate.
"""
from dataclasses import dataclass
import numpy as np

C = 3e8


@dataclass
class Params:
    fc: float = 28e9
    df: float = 120e3
    n_sub: int = 256
    n_sym: int = 128
    cp_ratio: float = 0.07

    @property
    def t_sym(self):
        return (1 + self.cp_ratio) / self.df

    @property
    def range_res(self):
        return C / (2 * self.n_sub * self.df)

    @property
    def vel_res(self):
        return C / (2 * self.fc * self.n_sym * self.t_sym)

    @property
    def range_max(self):
        return C / (2 * self.df)

    @property
    def vel_max(self):
        return C / (4 * self.fc * self.t_sym)


def qam(order, shape, rng):
    side = int(np.sqrt(order))
    lv = 2 * np.arange(side) - side + 1
    i = rng.integers(0, side, shape)
    q = rng.integers(0, side, shape)
    s = (lv[i] + 1j * lv[q]) / np.sqrt(2 * (order - 1) / 3)
    return s


def echo(p, targets, tx, snr_db, rng):
    """targets: list of (range m, speed m/s, amplitude). snr_db is per resource element."""
    k = np.arange(p.n_sub)[:, None]
    m = np.arange(p.n_sym)[None, :]
    h = np.zeros((p.n_sub, p.n_sym), complex)
    for r, v, a in targets:
        h += a * np.exp(-2j * np.pi * k * p.df * 2 * r / C) * \
            np.exp(2j * np.pi * m * p.t_sym * 2 * v * p.fc / C)
    sigma = 10 ** (-snr_db / 20)
    n = sigma * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)) / np.sqrt(2)
    return h * tx + n


def pilot_mask(p, spacing):
    mask = np.zeros((p.n_sub, p.n_sym), bool)
    mask[::spacing, :] = True
    return mask


def range_doppler(p, rx, tx, mask=None):
    est = rx / tx
    if mask is not None:
        est = est * mask
    win = np.hanning(p.n_sub)[:, None] * np.hanning(p.n_sym)[None, :]
    rd = np.fft.ifft(est * win, axis=0)
    rd = np.fft.fftshift(np.fft.fft(rd, axis=1), axes=1)
    return np.abs(rd) ** 2


def cfar(rd, guard=3, train=8, pfa=1e-5):
    """2-D cell-averaging CFAR."""
    n = (2 * (guard + train) + 1)
    kernel = np.ones((n, n))
    kernel[train:-train, train:-train] = 0
    count = kernel.sum()
    # The map is periodic in range and speed, so the training window wraps around.
    pad = np.zeros_like(rd)
    pad[:n, :n] = kernel
    pad = np.roll(pad, (-(n // 2), -(n // 2)), axis=(0, 1))
    noise = np.maximum(np.fft.ifft2(np.fft.fft2(rd) * np.fft.fft2(pad)).real, 1e-12) / count
    alpha = count * (pfa ** (-1 / count) - 1)
    return rd > alpha * noise


def cell_of(p, r, v):
    return int(round(r / p.range_res)) % p.n_sub, int(round(v / p.vel_res)) + p.n_sym // 2


def refine_range(rd, cell, p):
    """Parabolic interpolation around the peak, in metres."""
    k, m = cell
    window = rd[[(k - 1) % p.n_sub, k, (k + 1) % p.n_sub], m]
    a, b, c = np.log(window + 1e-20)
    return (k + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)) * p.range_res


def ber_awgn(p, order, snr_db, rng, n=200000):
    s = qam(order, n, rng)
    sigma = 10 ** (-snr_db / 20)
    y = s + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    side = int(np.sqrt(order))
    scale = np.sqrt(2 * (order - 1) / 3)
    idx = lambda x: np.clip(np.round((x * scale + side - 1) / 2), 0, side - 1).astype(int)
    return float(np.mean((idx(y.real) != idx(s.real)) | (idx(y.imag) != idx(s.imag))))
