# -*- coding:utf-8 -*-
# @Time: 2023/6/1 15:50
import numpy as np
from qam_modulator import qam_mod, draw_mod_constellation, qam_mapper, draw_trx_constellation, channel_Rician,channel_Awgn, qam_demapper,  bit_error_rate, draw_ber_curve, draw_ber_curve_smooth

# Settings
seed = 0  # random seed
np.random.seed(seed)
M = 16  # modulation order
snr = 20  # signal-to-noise ratio
snr_range = np.arange(0, 20, 0.1)  # test SNR range
channel = 'rician'  # channel type
tx_bits = np.random.randint(0, 2, 10000)

# M-QAM Modulation
mapping_table, demapping_table = qam_mod(M)
draw_mod_constellation(mapping_table)
tx_symbols = qam_mapper(tx_bits, mapping_table)
draw_trx_constellation(tx_symbols, tx=True)

# Wireless Channel
rx_symbols = channel_Awgn(tx_symbols, snr=snr)
# rx_symbols = channel_Rayleigh(tx_symbols, snr=snr)
# rx_symbols = channel_Rician(tx_symbols, snr=snr)
draw_trx_constellation(rx_symbols, tx=False, snr=snr, channel=channel)

# M-QAM Demodulation
rx_bits = qam_demapper(rx_symbols, demapping_table)
rx_bits = rx_bits[: len(tx_bits)]

# # Calculate BER
ber = bit_error_rate(tx_bits, rx_bits)
print(f"Bit Error Rate: {ber}")

# Draw BER Curve
draw_ber_curve(tx_bits, M, snr_range, channel='awgn')

# Draw smothed BER Curve
draw_ber_curve_smooth(tx_bits, M, snr_range, channel='awgn')

print('Completed.')
