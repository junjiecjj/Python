#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:25:48 2024

@author: jack

https://blog.csdn.net/weixin_43935696/article/details/120596442



"""


####################################  1  ##########################################################

import numpy as np
import commpy as cpy


## 调制
bits = np.random.binomial(n=1, p=0.5, size=(128))
Modulation_type ="BPSK"
if Modulation_type=="BPSK":
    bpsk = cpy.PSKModem(2)
    symbol = bpsk.modulate(bits)
    # return symbol
elif Modulation_type=="QPSK":
    qpsk = cpy.PSKModem(4)
    symbol = qpsk.modulate(bits)
    # return symbol
elif Modulation_type=="8PSK":
    psk8 = cpy.PSKModem(8)
    symbol = psk8.modulate(bits)
    # return symbol
elif Modulation_type=="16QAM":
    qam16 = cpy.QAMModem(16)
    symbol = qam16.modulate(bits)
    # return symbol
elif Modulation_type=="64QAM":
    qam64 = cpy.QAMModem(64)
    symbol = qam64.modulate(bits)
    # return symbol


# 解调, 和调制一样，需要先定义调制方法的类，再去调用解调的函数。
import commpy as cpy


bits = np.random.binomial(n=1,p=0.5,size=(20))
# Modem : QPSK
modem = cpy.QAMModem(16)
modem.plot_constellation()

signal = modem.modulate(bits)
rec_bits = modem.demodulate(signal, 'soft', noise_var = 40)


print(f"bits = \n {bits}")
print(f"rec_bits = \n {rec_bits}")




####################################  2  ##########################################################
import math

import matplotlib.pyplot as plt
import numpy as np

import commpy.channelcoding.convcode as cc
import commpy.channels as chan
import commpy.links as lk
import commpy.modulation as mod
import commpy.utilities as util


# =============================================================================
# Convolutional Code 1: G(D) = [1+D^2, 1+D+D^2]
# Standard code with rate 1/2
# =============================================================================

# Number of delay elements in the convolutional encoder
memory = np.array(2, ndmin=1)

# Generator matrix
g_matrix = np.array((0o5, 0o7), ndmin=2)

# Create trellis data structure
trellis1 = cc.Trellis(memory, g_matrix)

# =============================================================================
# Convolutional Code 1: G(D) = [1+D^2, 1+D^2+D^3]
# Standard code with rate 1/2
# =============================================================================

# Number of delay elements in the convolutional encoder
memory = np.array(3, ndmin=1)

# Generator matrix (1+D^2+D^3 <-> 13 or 0o15)
g_matrix = np.array((0o5, 0o15), ndmin=2)

# Create trellis data structure
trellis2 = cc.Trellis(memory, g_matrix)

# =============================================================================
# Convolutional Code 2: G(D) = [[1, 0, 0], [0, 1, 1+D]]; F(D) = [[D, D], [1+D, 1]]
# RSC with rate 2/3
# =============================================================================

# Number of delay elements in the convolutional encoder
memory = np.array((1, 1))

# Generator matrix & feedback matrix
g_matrix = np.array(((1, 0, 0), (0, 1, 3)))
feedback = np.array(((2, 2), (3, 1)))

# Create trellis data structure
trellis3 = cc.Trellis(memory, g_matrix, feedback, 'rsc')

# =============================================================================
# Basic example using homemade counting and hard decoding
# =============================================================================

# Traceback depth of the decoder
# tb_depth = None  # Default value is 5 times the number or memories

# for trellis in (trellis1, trellis2, trellis3):
#     for i in range(10):
#         # Generate random message bits to be encoded
#         message_bits = np.random.randint(0, 2, 1000)

#         # Encode message bits
#         coded_bits = cc.conv_encode(message_bits, trellis)

#         # Introduce bit errors (channel)
#         coded_bits[np.random.randint(0, 1000)] = 0
#         coded_bits[np.random.randint(0, 1000)] = 0
#         coded_bits[np.random.randint(0, 1000)] = 1
#         coded_bits[np.random.randint(0, 1000)] = 1

#         # Decode the received bits
#         decoded_bits = cc.viterbi_decode(coded_bits.astype(float), trellis, tb_depth)

#         num_bit_errors = util.hamming_dist(message_bits, decoded_bits[:len(message_bits)])

#         if num_bit_errors != 0:
#             print(num_bit_errors, "Bit Errors found!")
#         elif i == 9:
#             print("No Bit Errors :)")

# print("ssssssssssssssssssssssssssssssss")
# ==================================================================================================
# Complete example using Commpy features and compare hard and soft demodulation. Example with code 1
# ==================================================================================================

# Modem : QPSK
modem = mod.QAMModem(4)

# AWGN channel
channels = chan.SISOFlatChannel(None, (1 + 0j, 0j))

# SNR range to test
SNRs = np.arange(0, 6) + 10 * math.log10(modem.num_bits_symbol)

# Modulation function
def modulate(bits):
    return modem.modulate(cc.conv_encode(bits, trellis1, 'cont'))

# Receiver function (no process required as there are no fading)
def receiver_hard(y, h, constellation, noise_var):
    return modem.demodulate(y, 'hard')

# Receiver function (no process required as there are no fading)
def receiver_soft(y, h, constellation, noise_var):
    return modem.demodulate(y, 'soft', noise_var)

# Decoder function
def decoder_hard(msg):
    return cc.viterbi_decode(msg, trellis1)

# Decoder function
def decoder_soft(msg):
    return cc.viterbi_decode(msg, trellis1, decoding_type='soft')


# Build model from parameters
code_rate = trellis1.k / trellis1.n
model_hard = lk.LinkModel(modulate, channels, receiver_hard, modem.num_bits_symbol, modem.constellation, modem.Es, decoder_hard, code_rate)
model_soft = lk.LinkModel(modulate, channels, receiver_soft, modem.num_bits_symbol, modem.constellation, modem.Es, decoder_soft, code_rate)

# Test
BERs_hard = model_hard.link_performance(SNRs, 10000, 600, 5000, code_rate)
BERs_soft = model_soft.link_performance(SNRs, 10000, 600, 5000, code_rate)
plt.semilogy(SNRs, BERs_hard, 'o-', SNRs, BERs_soft, 'o-')
plt.grid()
plt.xlabel('Signal to Noise Ration (dB)')
plt.ylabel('Bit Error Rate')
plt.legend(('Hard demodulation', 'Soft demodulation'))
out_fig = plt.gcf()

out_fig.savefig('/home/jack/snap/MQAM.eps',   bbox_inches = 'tight')
plt.close()


























