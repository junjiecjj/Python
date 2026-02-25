import scipy
import numpy as np

p_freq_path = "data/p_freq_255.mat"
r_max = 200
v_max = 60
N_prbs = 255
N_sym = 256
matrix_size = (N_prbs, N_sym)

fc = 24e9
T_d = 5.1e-6
T_chip = T_d / (2 * N_prbs)
T_prbs = T_d / 2
T_D = T_d * N_sym
T_frame = T_D / 2
fs = 1 / T_chip
B = fs
range_axis = np.linspace(0, r_max, N_prbs)
doppler_freq = np.linspace(-1 / (T_d * 2), 1 / (T_d * 2), N_sym)
speed_axis = (doppler_freq * scipy.constants.speed_of_light) / (2 * fc)
