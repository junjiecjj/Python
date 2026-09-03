#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic matrix verification for
"通信中心ISAC感知接收机完整数学模型".

Verified identities:
1. x_tilde = Rt*Pt*GammaQ*Acp*U*s = Pt_cir*GammaQN*U*s;
2. Rs*Jtilde_tau*x_t = J_tau*x_tilde;
3. y_s = sum_q beta_q*J_tau_q*x_tilde;
4. Periodic cross-correlation computed by matrices equals its FFT implementation;
5. The matched-filter output equals shifted copies of the periodic ACF.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
np.random.seed(42)

def FFTmatrix(N):
    mat = np.zeros((N,N),dtype=complex)
    nn = np.arange(N)
    for k in range(N):
        mat[k,:] = np.exp(-1j*2*np.pi*k*nn/N)/np.sqrt(N)
    return mat


def AcpMat(N,Ncp):
    Acp = np.block([[np.zeros((Ncp,N-Ncp)),np.eye(Ncp)],[np.eye(N)]])
    return Acp


def upsamplingMatrix(inputLength,Q):
    GammaQ = np.zeros((Q*inputLength,inputLength))
    GammaQ[0::Q,:] = np.eye(inputLength)
    return GammaQ


def convMatrix(h,inputLength):
    h = np.asarray(h,dtype=complex).reshape(-1)
    col = np.hstack((h,np.zeros(inputLength-1,dtype=complex)))
    row = np.hstack((h[0],np.zeros(inputLength-1,dtype=complex)))
    H = toeplitz(col,row)
    return H


def circulantConvMatrix(h,N):
    hPeriodic = np.zeros(N,dtype=complex)
    for indexSample,value in enumerate(np.asarray(h,dtype=complex).reshape(-1)):
        hPeriodic[indexSample%N] += value
    Hcir = np.zeros((N,N),dtype=complex)
    for indexColumn in range(N):
        Hcir[:,indexColumn] = np.roll(hPeriodic,indexColumn)
    return Hcir


def linearDelayMatrix(outputLength, inputLength, delay):
    Jtilde = np.zeros((outputLength,inputLength))
    Jtilde[delay:delay+inputLength,:] = np.eye(inputLength)
    return Jtilde


def circularDelayMatrix(N, delay):
    delay = delay%N
    J = np.zeros((N,N))
    J[0:delay,N-delay:N] = np.eye(delay)
    J[delay:N,0:N-delay] = np.eye(N-delay)
    return J


def relativeError(left,right):
    return np.linalg.norm(left-right)/max(np.linalg.norm(right),np.finfo(float).eps)



# System dimensions
N = 32
Ncp = 16
Q = 4
M = N+Ncp
betaPulse = 0.35
targetDelay = np.array([20, 45], dtype = int)
targetCoefficient = np.array([1, 0.3], dtype = complex)
numberTarget = targetDelay.size
tauMax = np.max(targetDelay)

# A deterministic length-9 pulse; only its length and linear-convolution role are needed for the identities.
p_t = np.array([0.03,0.10,0.24,0.43,0.56,0.43,0.24,0.10,0.03], dtype=complex)
p_t = p_t/np.linalg.norm(p_t)
Lp = p_t.size

if Q*Ncp < Lp-1+tauMax:
    raise ValueError('The high-rate CP must satisfy Q*Ncp >= Lp-1+tauMax.')

# Dimensions appearing in the Markdown model
Kt = Q*M+Lp-1
Ks = Kt+tauMax

# Deterministic QPSK communication symbols and modulation basis x=U*s
symbolIndex = np.random.randint(0,4,N)
s = np.exp(1j*np.pi/2*symbolIndex)
F = FFTmatrix(N)
U = F.conj().T
x = U@s

# x_cp=Acp*x, x_up=GammaQ*x_cp, x_t=Pt*x_up
Acp = AcpMat(N, Ncp)
GammaQ = upsamplingMatrix(M, Q)
Pt = convMatrix(p_t, Q*M)
x_cp = Acp@x
x_up = GammaQ@x_cp
x_t = Pt@x_up

# Rt extracts the length-QN pulse-shaped useful transmit block.
Rt = np.block([np.zeros((Q*N, Q*Ncp)), np.eye(Q*N), np.zeros((Q*N, Lp-1))])
x_tilde = Rt@x_t

# Verify that physical linear pulse shaping becomes periodic pulse shaping on the useful block.
GammaQN = upsamplingMatrix(N, Q)
Pt_cir = circulantConvMatrix(p_t, Q*N)
x_tilde_equivalent = Pt_cir@GammaQN@x
pulseCircularizationMatrixError = relativeError(Rt@Pt@GammaQ@Acp, Pt_cir@GammaQN)
pulseCircularizationSignalError = relativeError(x_tilde, x_tilde_equivalent)

# Rs selects the same absolute observation interval from the longer full sensing echo.
Rs = np.block([np.zeros((Q*N,Q*Ncp)),np.eye(Q*N),np.zeros((Q*N,Lp-1+tauMax))])

# Construct every target echo first by a physical nonperiodic linear delay.
y_s_full_target = np.zeros((Ks, numberTarget), dtype = complex)
y_s_target = np.zeros((Q*N, numberTarget), dtype = complex)
targetCircularizationError = np.zeros(numberTarget)

for q in range(numberTarget):
    JtildeTau = linearDelayMatrix(Ks, Kt, targetDelay[q])
    JTau = circularDelayMatrix(Q*N,targetDelay[q])
    y_s_full_target[:,q] = targetCoefficient[q]*JtildeTau@x_t
    y_s_target[:,q] = Rs@y_s_full_target[:,q]
    y_s_target_equivalent = targetCoefficient[q]*JTau@x_tilde
    targetCircularizationError[q] = relativeError(y_s_target[:,q], y_s_target_equivalent)

y_s_full = np.sum(y_s_full_target,axis=1)
y_s = Rs@y_s_full
y_s_equivalent = np.sum(y_s_target,axis=1)
multiTargetSelectionError = relativeError(y_s, y_s_equivalent)

# Periodic sensing matched filter: matrix calculation for every candidate delay k.
periodicCrossCorrelationMatrix = np.zeros(Q*N, dtype=complex)
periodicACFMatrix = np.zeros(Q*N, dtype=complex)

for k in range(Q*N):
    JK = circularDelayMatrix(Q*N, k)
    periodicCrossCorrelationMatrix[k] = (JK@x_tilde).conj().T @ y_s
    periodicACFMatrix[k] = (JK@x_tilde).conj().T @ x_tilde

# Equivalent FFT implementations of the same periodic cross-correlation and ACF.
periodicCrossCorrelationFFT = np.fft.ifft(np.fft.fft(y_s)*np.conj(np.fft.fft(x_tilde)))
periodicACFFFT = np.fft.ifft(np.abs(np.fft.fft(x_tilde))**2)
crossCorrelationFFTError = relativeError(periodicCrossCorrelationMatrix, periodicCrossCorrelationFFT)
periodicACFFFTError = relativeError(periodicACFMatrix, periodicACFFFT)

# Verify y_tilde[k]=sum_q beta_q*R_x_tilde[k-tau_q].
periodicCrossCorrelationFromACF = np.zeros(Q*N,dtype=complex)
for q in range(numberTarget):
    periodicCrossCorrelationFromACF += targetCoefficient[q]*np.roll(periodicACFMatrix,targetDelay[q])
acfShiftSuperpositionError = relativeError(periodicCrossCorrelationMatrix,periodicCrossCorrelationFromACF)

# Dimension report
print('================ Matrix dimensions ================')
print(f's                 : {s.shape}')
print(f'U                 : {U.shape}')
print(f'Acp               : {Acp.shape}')
print(f'GammaQ            : {GammaQ.shape}')
print(f'Pt                : {Pt.shape}')
print(f'x_t               : {x_t.shape}')
print(f'Rt                : {Rt.shape}')
print(f'x_tilde           : {x_tilde.shape}')
print(f'Rs                : {Rs.shape}')
print(f'y_s_full          : {y_s_full.shape}')
print(f'y_s               : {y_s.shape}')

# Numerical identity report
print('\n================ Verification errors ================')
print(f'Pulse circularization matrix error       = {pulseCircularizationMatrixError:.3e}')
print(f'Pulse circularization signal error       = {pulseCircularizationSignalError:.3e}')
for q in range(numberTarget):
    print(f'Target {q+1} circularization error          = {targetCircularizationError[q]:.3e}')
print(f'Multi-target selection error             = {multiTargetSelectionError:.3e}')
print(f'Periodic cross-correlation FFT error     = {crossCorrelationFFTError:.3e}')
print(f'Periodic ACF FFT error                   = {periodicACFFFTError:.3e}')
print(f'ACF shifted-superposition error          = {acfShiftSuperpositionError:.3e}')


# First calculate the periodic ACF of x_tilde at every delay index, and then construct the range profile from shifted ACFs.
delayIndex = np.arange(-Q*N//2,Q*N//2)
periodicACF = periodicACFMatrix.copy()
targetRangeProfile = np.zeros((numberTarget,Q*N),dtype=complex)
for q in range(numberTarget):
    targetRangeProfile[q,:] = targetCoefficient[q]*np.roll(periodicACF,targetDelay[q])
rangeProfileComplex = np.sum(targetRangeProfile,axis=0)
rangeProfileFromReceivedSignal = periodicCrossCorrelationMatrix.copy()
rangeProfileConstructionError = relativeError(rangeProfileComplex,rangeProfileFromReceivedSignal)

# Use the same centered delay ordering as the Iceberg-curve reproductions.
periodicACFShifted = np.fft.fftshift(periodicACF)
targetRangeProfileShifted = np.fft.fftshift(targetRangeProfile,axes=1)
rangeProfileComplexShifted = np.fft.fftshift(rangeProfileComplex)
normalization = np.max(np.abs(rangeProfileComplexShifted)**2)
periodicACFdB = 10*np.log10(np.abs(periodicACFShifted)**2/normalization+1e-14)
targetRangeProfiledB = 10*np.log10(np.abs(targetRangeProfileShifted)**2/normalization+1e-14)
rangeProfiledB = 10*np.log10(np.abs(rangeProfileComplexShifted)**2/normalization+1e-14)

print(f'Range-profile construction error          = {rangeProfileConstructionError:.3e}')

# Plot every shifted target ACF and their coherent sum.
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
fig,axs = plt.subplots(1,1,figsize=(8,6),constrained_layout=True)
targetColor = ['#F65314','#8A2BE2','#05C349','#FFBB00']
for q in range(numberTarget):
    axs.plot(delayIndex, targetRangeProfiledB[q,:],color=targetColor[q%len(targetColor)],linestyle='-',linewidth=2,label=f'Target {q+1}')
axs.plot(delayIndex,rangeProfiledB,color='#00A1F1',linestyle='--',linewidth=1,label='Range Profile', marker = 'o', markevery = 8, mfc = 'none', ms = 10)
axs.set_xlabel('Delay Index')
axs.set_ylabel('Normalized Correlation Level (dB)')
axs.set_xlim([-Q*N//2,Q*N//2])
# axs.set_ylim([-80,0])
axs.set_xticks(np.arange(-Q*N//2,Q*N//2+1,16))
# axs.set_yticks(np.arange(-80,1,10))
axs.grid(True,linestyle='--',alpha=0.3)
axs.legend(loc='best',edgecolor='black')
# plt.savefig('ISAC_Sensing_Periodic_Correlation.pdf',bbox_inches='tight')
# plt.savefig('ISAC_Sensing_Periodic_Correlation.png',dpi=300,bbox_inches='tight')
plt.show()
plt.close()

















