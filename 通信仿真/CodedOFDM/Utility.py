#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:53:35 2024

@author: jack
"""



import numpy as np



def ofdm_tx(x, nfft, nsc, cp_length):
    """ OFDM Transmit signal generation """

    nfft = float(nfft)
    nsc = float(nsc)
    cp_length = float(cp_length)
    ofdm_tx_signal = np.array([])

    for i in range(0, np.shape(x)[1]):
        symbols = x[:, i]
        ofdm_sym_freq = np.zeros(nfft, dtype=complex)
        ofdm_sym_freq[1:(nsc / 2) + 1] = symbols[nsc / 2:]
        ofdm_sym_freq[-(nsc / 2):] = symbols[0:nsc / 2]
        ofdm_sym_time = np.fft.ifft(ofdm_sym_freq)
        cp = ofdm_sym_time[-cp_length:]
        ofdm_tx_signal = np.concatenate((ofdm_tx_signal, cp, ofdm_sym_time))

    return ofdm_tx_signal


def ofdm_rx(y, nfft, nsc, cp_length):
    """ OFDM Receive Signal Processing """

    num_ofdm_symbols = int(len(y) / (nfft + cp_length))
    x_hat = np.zeros([nsc, num_ofdm_symbols], dtype=complex)

    for i in range(0, num_ofdm_symbols):
        ofdm_symbol = y[i * nfft + (i + 1) * cp_length:(i + 1) * (nfft + cp_length)]
        symbols_freq = np.fft.fft(ofdm_symbol)
        x_hat[:, i] = np.concatenate((symbols_freq[-nsc / 2:], symbols_freq[1:(nsc / 2) + 1]))

    return x_hat







    def plot_constellation(self):
        """ Plot the constellation """
        plt.figure(figsize=(2, 2)) # 6，8分别对应宽和高
        plt.scatter(self.constellation.real, self.constellation.imag)

        # bin(i)[2:].rjust(nbits, '0')
        for symb in self.constellation:
            # print(len(self.demodulate(symb, 'hard')))
            plt.text(symb.real, symb.imag + .2, "".join([str(i) for i in self.demodulate(symb, 'hard')]), ha='center')
            plt.text(symb.real, symb.imag - .3, symb, ha='center')

        plt.title('Constellation')
        plt.grid()
        plt.show()











































