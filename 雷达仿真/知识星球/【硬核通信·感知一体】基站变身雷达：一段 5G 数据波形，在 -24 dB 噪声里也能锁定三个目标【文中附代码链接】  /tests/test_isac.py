import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from isac import *


def test_target_appears_in_right_cell():
    p = Params()
    rng = np.random.default_rng(1)
    tx = qam(16, (p.n_sub, p.n_sym), rng)
    rd = range_doppler(p, echo(p, [(50.0, 10.0, 1.0)], tx, 20, rng), tx)
    k, m = np.unravel_index(np.argmax(rd), rd.shape)
    ck, cm = cell_of(p, 50.0, 10.0)
    assert abs(k - ck) <= 1 and abs(m - cm) <= 1


def test_qam_has_unit_power():
    s = qam(16, 200000, np.random.default_rng(2))
    assert abs(np.mean(np.abs(s) ** 2) - 1) < 0.01


def test_noise_only_gives_few_false_alarms():
    p = Params()
    rng = np.random.default_rng(3)
    n = (rng.standard_normal((p.n_sub, p.n_sym)) + 1j * rng.standard_normal((p.n_sub, p.n_sym))) / np.sqrt(2)
    win = np.hanning(p.n_sub)[:, None] * np.hanning(p.n_sym)[None, :]
    assert cfar(np.abs(np.fft.fft2(n * win)) ** 2).sum() < 5
