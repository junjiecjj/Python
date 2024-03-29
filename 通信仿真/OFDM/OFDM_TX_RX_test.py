

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jan  7 17:48:09 2024
@author: jack

这套代码有错误，很严重的错误，别看了。


https://github.com/Niroman/OFDM-Transmitter-and-Receiver

"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import random
import cmath
import math

'''
# OFDM Transmitter
#
# Generation of input bits at the transmitter
# 1. Payload Bits
# 2. Pilot Bits - It is the reference signals
# 3. Preamble

t=np.linspace(-0.02,0.05,1000)
plt.plot(t, 325 * np.sin(2*np.pi*50*t))
#plt.show()

sampling_freq = 3.84 #in Mhz
carrier_freq = 3.4 #in Ghz
subcarrier_space = 15 #in Khz
OFDM_symbol = 256 # Number of samples
active_sub = 179
bandwidth = 2.7 #Mhz
cyclic_prefix = 18
#Data information bits are randomly generated. Payload is QPSk

#preamble follows Zadoff-chu seq
Nzc = 839
u = 2
cv = 0
Ncs = 0
'''

#%% Pilot Generation, 生成导频
## 伪随机序列常用于生成一个特定的序列或干扰一个特定信道的数据。LTE中使用的伪随机序列为Gold序列。
###  https://zhuanlan.zhihu.com/p/491592801
pilotSymbol = np.zeros(4200, dtype = np.complex128)
pilotSymbolNumber = 0

for slotNumber in range(0, 20):
    for symNumber in range(0, 7):
        # 1st initialized sequence m
        x1_init = np.zeros((31))
        x1_init[0] = 1
        # Second Initial Condition of the polynomail x2() depending on the sequence
        x2_init = np.zeros((31))
        x2_init[0] = 1
        x2_init[10] = 7 * (slotNumber + 1) + symNumber + 1

        # Mpn为最后序列c()的长度，自定义
        Mpn = 4500
        # Nc为固定值，ts 36.211中有定义
        Nc = 1600

        # 用0初始化x1(n)序列和x2(n)序列，序列长度为Nc + Mpn + 31
        x1 = np.zeros(Nc + Mpn + 31)
        x2 = np.zeros(Nc + Mpn + 31)
        # Create a vector for c() all initialized with 0
        c = np.zeros(Mpn)

        ## 初始化x1(n)序列和x2(n)序列
        x1[0:31] = x1_init
        x2[0:31] = x2_init

        ##  x1(n)序列的初始值，为固定值
        for n in range(0, Mpn + Nc):
            x1[n + 31] = (x1[n + 3] + x1[n]) % 2
        ## x2()序列的初始值c_init，由循环前缀CP的类型、小区ID、时隙号和OFDM符号的序号等参数共同决定
        for n in range(0, Mpn + Nc):
            x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) % 2

        # 用2个m序列生成Gold序列 : c()
        for n in range(0, Mpn):
            c[n] = (x1[n + Nc] + x2[n + Nc]) % 2

        # Reference symbol for each pilot
        for max_RB in range(0, 30):
            if max_RB > 0:
                real_num = ((1 / (np.sqrt(2))) * (1 - (2 * c[2 * max_RB])))
                imag_num = ((1 / (np.sqrt(2))) * (1 - (2 * c[2 * max_RB + 1])))
                complex_number = complex(real_num, imag_num)
            else:
                real_num = (1 / (np.sqrt(2)))
                imag_num = ((1 * (1 - (2 * c[1])))/ (np.sqrt(2)))
                complex_number = complex(real_num, imag_num)
            pilotSymbol[pilotSymbolNumber] = complex_number
            pilotSymbolNumber += 1

# print("Printing the Final Pilot Symbols for every Resource Blocks.",pilotSymbol)

# Checkpoint must generate the modulation of Payload Bits.

#%% Modulation of Payload Bits
# 50000 Payload bits randomly generated.
payloadBits = np.random.choice([0,1], size = 50000)

dataSymbol = np.zeros(int(len(payloadBits)/2), dtype = np.complex128)
dataSymBitNo = 0

for data_bit in range(0, len(payloadBits), 2):
    if ( payloadBits[data_bit] == 0 and payloadBits[data_bit + 1] == 0):
        dataSymbol[dataSymBitNo] = cmath.exp(1 * math.pi / 4j)
    elif ( payloadBits[data_bit] == 1 and payloadBits[data_bit + 1] == 0):
        dataSymbol[dataSymBitNo] = cmath.exp(1 * 3 * math.pi / 4j)
    elif ( payloadBits[data_bit] == 0 and payloadBits[data_bit + 1] == 1):
        dataSymbol[dataSymBitNo] = cmath.exp(1 * 7 * math.pi / 4j)
    elif ( payloadBits[data_bit] == 1 and payloadBits[data_bit + 1] == 1):
        dataSymbol[dataSymBitNo] = cmath.exp(1 * 5 * math.pi / 4j)
    dataSymBitNo +=1

# print("Printing the Modulated Data Symbols ",dataSymbol)


#%% Generating Modulator Output after FFT
Num_fft = 256
Subcarrier_Space = 15*1000 # 15Khz
Sampling_Frequency = Num_fft * Subcarrier_Space  # 采样频率，3.84Mhz = 256*15 Khz，
Sampling_Period = 1 / Sampling_Frequency         # 采样周期：相邻采样点的时间间隔
Sampled_data = np.fft.fft(dataSymbol) * Sampling_Period
Shifted_Data = np.fft.fftshift( Sampled_data )
Length_FFT_Data = len(Sampled_data)
Time_FFT = Sampling_Frequency/Length_FFT_Data
Sampled_Time = np.arange(0, Sampling_Frequency, Time_FFT) - Sampling_Frequency/2

#%% Preamble Generation; 前导符号（Preamble）
# 由于终端的移动性，终端和网络之间的距离是不确定的，所以如果终端需要发送消息到网络，则必须实时进行上行同步的维持管理。PRACH的目的就是为达到上行同步,建立和网络上行同步关系以及请求网络分配给终端专用资源，进行正常的业务传输。
Tx_preamble = np.zeros(839,dtype=complex)
Rx_preamble = np.zeros(181,dtype=complex)
for Nzc in range(len(Tx_preamble)):
    Tx_preamble[Nzc] = cmath.exp( -(math.pi * 2 * Nzc * (Nzc-1) / 839j))
    if Nzc < 181:
        Rx_preamble[Nzc] = Tx_preamble[Nzc]

#%% Multiplexing Preambles, Pilots and Payload Symbols
# Declaration of Variables

pilot_symbol_counter = 0 # To increment in generated Pilot Symbols one by one
data_symbol_counter = 0 # To increment in generated Data Symbols one by one

Average_equi_value = 0
phaseDelay = 2 + 3*random.random() # Generating Constant Phase Delay

demuxOutput = np.zeros(22000, dtype = complex) # Generating the Demultiplexed Output
demuxInBitNo = 0
fftPlotBits = np.zeros(len(payloadBits), dtype = complex) # To plot the data points, 50000
fftPlotBitNo = 0
equalPlotBits = np.zeros(len(payloadBits), dtype = complex) # To plot the equalizer points, 50000
equalPlotBitNo = 0
Extra_bit_gen = 3287 # To generate a currupt signal,


#%% Sending 140 OFDM Symbols
for ofdm_sym in range(0, 140):
    '''
    Transmitter Block
    '''
    ## Sending the data to Mux block
    pilot_6subcarrier_counter = 0 # To assign pilots to every 6th Sub Carrier
    mux_input = np.zeros(180, dtype = complex)

    ## Sending Preambles for Synchronization from the diagram
    if ofdm_sym < 2:
        for preamble_data in range(0, 180):
            mux_input[preamble_data] = Tx_preamble[preamble_data]
    else:
        for muxBit in range(0, 180):
            ## Pilot Subcarriers assignment
            if muxBit == pilot_6subcarrier_counter:
                mux_input[muxBit] = pilotSymbol[pilot_symbol_counter]
                pilot_symbol_counter += 1
                pilot_6subcarrier_counter += 6
            ## Dc Subcarrier Assignment
            elif muxBit == 128:
                mux_input[muxBit] = 0
            ## Data Subcarriers Assignment
            else:
                mux_input[muxBit] = dataSymbol[data_symbol_counter] # 180
                data_symbol_counter += 1

    ## Feeding the Mux outputs to IDFT block
    mux_Input_IDFT = np.zeros(256, dtype = complex)
    mux_Input_counter = 0

    for mux_Input_IDFT_bit in range(39, 218):
        mux_Input_IDFT[mux_Input_IDFT_bit] = mux_input[mux_Input_counter]
        mux_Input_counter += 1

    ## Applying IDFT to the Mux data and plotting in time domain signal
    ## Sampling_Frequency_IDFT = 3.84
    Output_IFFT = np.fft.ifft(mux_Input_IDFT)   ## 256

    ## Adding Cyclic Prefix to avoid ISI by taking the last 17 Values from IFFT
    Cyclic_Prefix = Output_IFFT[len(Output_IFFT) - 17 : ]
    Output_IFFT_CP_added = np.hstack((Cyclic_Prefix, Output_IFFT))  # 273 = 256 + 17

    Tx_signal_channel = Output_IFFT_CP_added

    '''
    Channel Block
    '''
    Delay_Signal_channel = Tx_signal_channel * np.exp(phaseDelay*1j) # Adding Constant Phase Delay to thr transmitted signal

    # AWGN Generation
    Target_SNR = 0.001  # 15dB
    Energy_of_Signal_beforeNoise = (np.sum(np.square(np.abs(Delay_Signal_channel))))/len(Delay_Signal_channel)

    # Finding the Noise Spectral Density of the Signal
    noiseSigma = np.sqrt((Energy_of_Signal_beforeNoise)/(2*Target_SNR))

    # AWGN Noise Generation to add to the generated signal
    AWGN_Noise = noiseSigma*((np.random.randn(len(Delay_Signal_channel))) + np.random.randn(len(Delay_Signal_channel))*1j)

    # Adding Noise to the Signal
    Output_Signal_after_noise = Delay_Signal_channel + AWGN_Noise # 273

    ## Extra bit generation to corrupt the signal
    Extra_Data = 0.1 * (np.random.randn(Extra_bit_gen) + np.random.randn(Extra_bit_gen) *1j) #  3287

    ## Adding Currupt signal to the Payload signal and plotting
    Output_Signal_after_noise_currupt = np.hstack((Extra_Data, Output_Signal_after_noise)) #  3560 = 3287+ 273

    '''
    Receiver Block
    '''

    '''
    Method - 1 To correlate and find the starting point of the signal from the currupt signal
    '''
    if ofdm_sym == 0:
        # Padding_Zeroes = np.zeros(len(Output_Signal_after_noise_currupt)-len(Output_IFFT),dtype=complex)*1j
        # Output_IFFT_padded = np.hstack((Output_IFFT ,Padding_Zeroes))

        Correlator_output = sig.correlate(Output_Signal_after_noise_currupt, Output_IFFT, mode='same') #  3560
        Abs_correlator_output = np.abs(Correlator_output)
        Maximum_abs_correlator = (np.max(Abs_correlator_output))
        # Lag = np.arange(0, len(Correlator_output), 1)   # Correlator_output.shape = 3560
        Preamble_start = np.where(Abs_correlator_output == Maximum_abs_correlator)

    # Removing CP and Extraneous Data
    # Must make some changes and find the error signals
    Output_Signal_after_CP_Noise_Removal = Output_Signal_after_noise_currupt[Preamble_start[0][0] - 128:]

    '''
    Method - 2 To correlate and find the starting point of the signal from the currupt signal

    if ofdm_sym == 0:
        # Padding_Zeroes = np.zeros(len(Output_Signal_after_noise_currupt)-len(Output_IFFT),dtype=complex)*1j
        # Output_IFFT_padded = np.hstack((Output_IFFT ,Padding_Zeroes))

        Correlator_output = sig.correlate(Output_Signal_after_noise_currupt, Output_IFFT, mode='full')
        Abs_correlator_output = np.abs(Correlator_output)
        Maximum_abs_correlator = (np.max(Abs_correlator_output))
        Lag = np.arange(0, len(Correlator_output), 1)
        Preamble_start = np.where(Abs_correlator_output == Maximum_abs_correlator)

        # Plotting Correlator O/P with Lag
        plt.figure(10)
        plt.stem(Lag, Correlator_output, use_line_collection = True)
        plt.title("Correlator and Lag plot")
        plt.xlabel("Sample Number(lag)")
        plt.ylabel("Correlator output")
        plt.show()

    ## Removing CP and Extraneous Data
    ## Must make some changes and find the error signals

    Output_Signal_after_CP_Noise_Removal = Output_Signal_after_noise_currupt[Preamble_start[0][0] - 255:]
    '''

    # Equalizing Output Plots
    if ofdm_sym > 1:
        # Performing DFT
        Output_Signal_after_Rx_FFT = np.fft.fft(Output_Signal_after_CP_Noise_Removal)

        equalizer = np.zeros(30, dtype = complex)
        equiNo = 0
        for equi in range(39, 218, 6):
            equalizer[equiNo] = Output_Signal_after_Rx_FFT[equi] / mux_Input_IDFT[equi]
            equiNo += 1

        # Average Alpha value calculation
        Average_equi_value = (np.angle(np.sum(equalizer)/len(equalizer)) + Average_equi_value)/2

        # Equalizer Output
        Equalizer_output = Output_Signal_after_Rx_FFT * (cmath.exp( -1j * Average_equi_value))  ## 256

        # Removing Pilot Bits and DC Carrier
        outQPSKPilot = np.zeros(30, dtype = complex)
        pilotBitNo = 39
        pilot = 0
        outQPSKData = np.zeros(149, dtype = complex)
        dataBitNo = 0

        for outBitNo in range(39, 218):
            if (outBitNo == pilotBitNo):
                outQPSKPilot[pilot] = Equalizer_output[outBitNo]
                pilotBitNo = pilotBitNo + 6
                pilot = pilot + 1
            elif(outBitNo != 128):
                outQPSKData[dataBitNo] = Equalizer_output[outBitNo]
                dataBitNo = dataBitNo + 1

        # Collecting Data Bits
        for demodInput in range (0, 149):
            demuxOutput[demuxInBitNo] = outQPSKData[demodInput]
            demuxInBitNo = demuxInBitNo + 1

#%% Demodulation of Payload
Final_Payload_at_RX = np.zeros(2* len(demuxOutput))
for demodBitNo in range(len(demuxOutput)-1):
    # I Value
    if (np.real(demuxOutput[demodBitNo]) > 0):
        Final_Payload_at_RX[2 * demodBitNo - 1] = 0
    else:
        Final_Payload_at_RX[2 * demodBitNo - 1] = 1
    # Q Value
    if (np.imag(demuxOutput[demodBitNo]) > 0):
        Final_Payload_at_RX[2 * demodBitNo] = 0
    else:
        Final_Payload_at_RX[2 * demodBitNo] = 1


#%% Final Payload
Final_Payload_at_TX = np.zeros(len(Final_Payload_at_RX))
for final_bit in range(len(Final_Payload_at_RX)):
    Final_Payload_at_TX[final_bit] = payloadBits[final_bit]

#%% Bit Error Calculation
BER = np.abs((np.sum(Final_Payload_at_RX - Final_Payload_at_TX))/len(demuxOutput))

print("Bit Error Rate after passing through the Channel : ",BER)


















