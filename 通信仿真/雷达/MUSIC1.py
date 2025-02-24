




# https://zhuanlan.zhihu.com/p/678205710

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitted signal
t = np.arange(N)/sample_rate
f_tone = 0.02e6
tx = np.exp(2j*np.pi*f_tone*t)

# Simulate three omnidirectional antennas in a line with 1/2 wavelength between adjancent ones, receiving a signal that arrives at an angle

d = 0.5
Nr = 3
theta_degrees = 20 # direction of arrival
theta = theta_degrees / 180 * np.pi # convert to radians
a = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta))
print(a)

# we have to do a matrix multiplication of a and tx, so first lets convert both to matrix' instead of numpy arrays which dont let us do 1d matrix math
a = np.asmatrix(a)
tx = np.asmatrix(tx)

# so how do we use this? simple:

r = a.T @ tx  # matrix multiply. dont get too caught up by the transpose a, the important thing is we're multiplying the array factor by the tx signal
print(r.shape) # r is now going to be a 2D array, 1d is time and 1d is spatial

# Plot the real part of the first 200 samples of all three elements

fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
ax1.set_ylabel("Samples")
ax1.set_xlabel("Time")
ax1.grid()
ax1.legend(['0','1','2'], loc=1)
plt.show()

# note the phase shifts, they are also there on the imaginary portions of the samples

# So far this has been simulating the recieving of a signal from a certain angle of arrival
# in your typical DOA problem you are given samples and have to estimate the angle of arrival(s)
# there are also problems where you have multiple receives signals from different directions and one is the SOI while another might be a jammer or interferer you have to null out

# One thing we didnt both doing- lets add noise to this recieved signal.
# AWGN with a phase shift applied is still AWGN so we can add it after or before the array factor is applied, doesnt really matter, we'll do it after
# we need to make sure each element gets an independent noise signal added

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.1*n


fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
ax1.plot(np.asarray(r[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
ax1.plot(np.asarray(r[1,:]).squeeze().real[0:200])
ax1.plot(np.asarray(r[2,:]).squeeze().real[0:200])
ax1.set_ylabel("Samples")
ax1.set_xlabel("Time")
ax1.grid()
ax1.legend(['0','1','2'], loc=1)
plt.show()


if True:
    Nr = 8 # 8 elements
    theta1 = 20 / 180 * np.pi # convert to radians
    theta2 = 25 / 180 * np.pi
    theta3 = -40 / 180 * np.pi
    a1 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta1)))
    a2 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta2)))
    a3 = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta3)))
    # we'll use 3 different frequencies
    r = a1.T @ np.asmatrix(np.exp(2j*np.pi*0.01e6*t)) + \
        a2.T @ np.asmatrix(np.exp(2j*np.pi*0.02e6*t)) + \
        0.1 * a3.T @ np.asmatrix(np.exp(2j*np.pi*0.03e6*t))
    n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
    r = r + 0.04*n

    # MUSIC Algorithm (part that doesn't change with theta_i)
    num_expected_signals = 3 # Try changing this!
    R = r @ r.H # Calc covariance matrix, it's Nr x Nr
    w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]

    if False:
        fig, (ax1) = plt.subplots(1, 1, figsize=(7, 3))
        ax1.plot(10*np.log10(np.abs(w)),'.-')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue [dB]')
        plt.show()
        #fig.savefig('../_images/doa_eigenvalues.svg', bbox_inches='tight') # I EDITED THIS ONE IN INKSCAPE!!!
        exit()

    eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
    v = v[:, eig_val_order] # sort eigenvectors using this order
    V = np.asmatrix(np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64)) # Noise subspace is the rest of the eigenvalues
    for i in range(Nr - num_expected_signals):
        V[:, i] = v[:, i]

    theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 100 different thetas between -180 and +180 degrees
    results = []
    for theta_i in theta_scan:
        a = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))) # look familiar?
        a = a.T
        metric = 1 / (a.H @ V @ V.H @ a) # The main MUSIC equation
        metric = np.abs(metric[0,0]) # take magnitude
        metric = 10*np.log10(metric) # convert to dB
        results.append(metric)

    results /= np.max(results) # normalize

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlabel_position(30)  # Move grid labels away from other labels
    plt.show()
    #fig.savefig('../_images/doa_music.svg', bbox_inches='tight')

    # exit()
