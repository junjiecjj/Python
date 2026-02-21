import numpy as np
import scipy.constants
import scipy.io

from utils.scales import scale_complex_matrices
from utils.plots import plot_rd
from utils.utils import freq_to_time
from configs.data import root_dir

# Load data from .mat file

mat_Y_PRBS_waveform = scipy.io.loadmat(f"{root_dir}/Y_PRBS_waveform/20/2.mat")
mat_PRBS_waveform = scipy.io.loadmat("data/p_freq_255.mat")


PRBS_waveform = mat_PRBS_waveform["data"]
Y_PRBS_waveform = mat_Y_PRBS_waveform["data"]


# # Input list of matrices
a, b = 0, 1
Y_PRBS_waveform = scale_complex_matrices(Y_PRBS_waveform, a, b, scale_magnitude=True)
PRBS_waveform = scale_complex_matrices(PRBS_waveform, a, b, scale_magnitude=True)

Z_PRBS_waveform = Y_PRBS_waveform * np.conj(PRBS_waveform)  # Simulate prediction


# Compute Range-Doppler response
z_PRBS_waveform = freq_to_time(Z_PRBS_waveform, dim=0)
plot_rd(z_PRBS_waveform, domain="time")
