#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:52:44 2025

@author: jack
"""

import numpy as np
from scipy.special import comb

from scipy.special import comb, beta as beta_func


def cacfar(signal, Pfa, ref_num, guard_num):
    """
    CA-CFAR (Cell Averaging Constant False Alarm Rate) detector.
    Parameters:
    -----------
        signal : array_like
            Data of signal (include signal and noise)
        Pfa : float
            Probability of false alarm
        ref_num : int
            Number of reference cells
        guard_num : int
            Number of guard cells
    Returns:
    --------
        position : ndarray
            Positions of target
        threshold : ndarray
            CFAR threshold of input signal
        start_cell : int
            Start cell index
        stop_cell : int
            Stop cell index
    """
    position = []
    left_num = guard_num + ref_num
    start_cell = left_num + 1
    stop_cell = len(signal) - left_num
    N = 2 * ref_num
    alpha = N * (Pfa ** (-1 / N) - 1)

    threshold = np.zeros(stop_cell - start_cell + 1)

    for ii in range(start_cell, stop_cell + 1 ):
        # Get reference cells (excluding guard cells)
        left_data = signal[ii - left_num - 1 : ii - guard_num - 1]  # Python indexing starts at 0
        right_data = signal[ii + guard_num : ii + left_num]     # Python indexing starts at 0

        tmp_data = np.concatenate([left_data, right_data])
        tmp = np.mean(tmp_data) * alpha

        threshold[ii - start_cell] = tmp

        if tmp < signal[ii - 1]:  # Adjust for Python's 0-based indexing
            position.append(ii)

    return np.array(position), threshold, start_cell, stop_cell

def gocacfar(signal, Pfa, ref_num, guard_num):
    """
    GO-CFAR (Greatest Of Constant False Alarm Rate) detector

    Parameters:
        -----------
        signal : array_like
            Data of signal (include signal and noise)
        Pfa : float
            Probability of false alarm
        ref_num : int
            Number of reference cells
        guard_num : int
            Number of guard cells
    Returns:
        --------
        position : ndarray
            Positions of target
        threshold : ndarray
            CFAR threshold of input signal
        start_cell : int
            Start cell index
        stop_cell : int
            Stop cell index
    """
    def compute_pfa(alpha, N):
        """Compute probability of false alarm for given alpha and N"""
        # GO-CA CFAR formula
        part1 = (1 + alpha / (N / 2)) ** (-N / 2)
        part2 = (2 + alpha / (N / 2)) ** (-N / 2)
        part3 = 0

        for k in range(0, int(N/2)):
            # Using scipy's comb function for combination calculation
            part3 += comb(int(N/2) - 1 + k, k) * (2 + alpha / (N/2)) ** (-k)

        pfa = part1 - part2 * part3
        pfa = 2 * pfa
        return pfa

    def get_alpha(Pfa_set, N):
        """Find alpha using bisection method"""
        left_alpha = 0

        # Find initial bracket
        while True:
            right_alpha = left_alpha + 1
            this_pfa = compute_pfa(right_alpha, N)
            if this_pfa < Pfa_set:
                break
            left_alpha = right_alpha

        # Bisection method
        mid_alpha = 0.5 * (left_alpha + right_alpha)
        this_pfa = compute_pfa(mid_alpha, N)

        while abs(this_pfa - Pfa_set) > 0.000001 * Pfa_set:
            if this_pfa > Pfa_set:
                left_alpha = mid_alpha
            else:
                right_alpha = mid_alpha

            mid_alpha = 0.5 * (left_alpha + right_alpha)
            this_pfa = compute_pfa(mid_alpha, N)

        return mid_alpha

    position = []
    left_num = guard_num + ref_num
    start_cell = left_num + 1
    stop_cell = len(signal) - left_num
    N = 2 * ref_num
    alpha = get_alpha(Pfa, N)

    threshold = np.zeros(stop_cell - start_cell + 1)

    for ii in range(start_cell, stop_cell + 1):
        # Get left and right reference cells
        left_data = signal[ii - left_num - 1:ii - guard_num - 1]  # Python indexing starts at 0
        right_data = signal[ii + guard_num:ii + left_num]  # Python indexing starts at 0

        # GO-CFAR: take the greater of the two averages
        tmp_data = max(np.mean(left_data), np.mean(right_data))
        tmp = tmp_data * alpha

        threshold[ii - start_cell] = tmp

        if tmp < signal[ii - 1]:  # Adjust for Python's 0-based indexing
            position.append(ii)

    return np.array(position), threshold, start_cell, stop_cell


def oscfar(signal, Pfa, ref_num, guard_num, k):
    """
    OS-CFAR (Ordered Statistics Constant False Alarm Rate) detector

    Parameters:
    -----------
        signal : array_like
            Data of signal (include signal and noise)
        Pfa : float
            Probability of false alarm
        ref_num : int
            Number of reference cells
        guard_num : int
            Number of guard cells
        k : int
            The k-th ordered statistical value used for threshold estimation

    Returns:
    --------
        position : ndarray
            Positions of detected targets
        threshold : ndarray
            CFAR threshold of input signal
        start_cell : int
            Start cell index for detection
        stop_cell : int
            Stop cell index for detection
    """

    def get_alpha(Pfa_set, N, k_val):
        """Calculate alpha using midpoint method for OS-CFAR"""
        # Find initial bracket
        left_alpha = 0
        while True:
            right_alpha = left_alpha + 1
            # OS-CFAR Pfa formula
            this_pfa = k_val * comb(N, k_val) * beta_func(right_alpha + N - k_val + 1, k_val)
            if this_pfa < Pfa_set:
                break
            left_alpha = right_alpha

        # Refine using midpoint method
        mid_alpha = 0.5 * (left_alpha + right_alpha)
        this_pfa = k_val * comb(N, k_val) * beta_func(mid_alpha + N - k_val + 1, k_val)

        while abs(this_pfa - Pfa_set) > 0.000001 * Pfa_set:
            if this_pfa > Pfa_set:
                left_alpha = mid_alpha
            else:
                right_alpha = mid_alpha

            mid_alpha = 0.5 * (left_alpha + right_alpha)
            this_pfa = k_val * comb(N, k_val) * beta_func(mid_alpha + N - k_val + 1, k_val)

        return mid_alpha

    position = []
    left_num = guard_num + ref_num
    start_cell = left_num + 1
    stop_cell = len(signal) - left_num
    N = 2 * ref_num
    alpha = get_alpha(Pfa, N, k)

    threshold = np.zeros(stop_cell - start_cell + 1)

    for ii in range(start_cell, stop_cell + 1):
        # Get reference cells (excluding guard cells)
        left_ref = signal[ii - left_num - 1:ii - guard_num - 1]  # Python indexing
        right_ref = signal[ii + guard_num:ii + left_num]  # Python indexing

        tmp_data = np.concatenate([left_ref, right_ref])
        # Sort the reference cells and select the k-th value
        sorted_data = np.sort(tmp_data)
        # Note: k is 1-indexed in MATLAB, so we use k-1 for Python 0-indexing
        T = sorted_data[k - 1] * alpha

        threshold_idx = ii - start_cell
        threshold[threshold_idx] = T

        if T < signal[ii - 1]:  # -1 for Python indexing
            position.append(ii)

    return np.array(position), threshold, start_cell, stop_cell


def socacfar(signal, Pfa, ref_num, guard_num):
    """
    SOCA-CFAR (Smallest Of Cell Averaging Constant False Alarm Rate) detector

    Parameters:
    -----------
    signal : array_like
        Data of signal (include signal and noise)
    Pfa : float
        Probability of false alarm
    ref_num : int
        Number of reference cells
    guard_num : int
        Number of guard cells

    Returns:
    --------
    position : ndarray
        Positions of detected targets
    threshold : ndarray
        CFAR threshold of input signal
    start_cell : int
        Start cell index for detection
    stop_cell : int
        Stop cell index for detection
    """

    def compute_pfa(alpha, N):
        """Compute probability of false alarm for SOCA-CFAR"""
        part1 = (2 + alpha / (N / 2)) ** (-N / 2)
        part2 = 0

        for k in range(0, int(N/2)):
            part2 += comb(int(N/2) - 1 + k, k) * (2 + alpha / (N / 2)) ** (-k)

        pfa = 2 * part1 * part2
        return pfa

    def get_alpha(Pfa_set, N):
        """Calculate alpha using midpoint method for SOCA-CFAR"""
        # Find initial bracket
        left_alpha = 0
        while True:
            right_alpha = left_alpha + 1
            this_pfa = compute_pfa(right_alpha, N)
            if this_pfa < Pfa_set:
                break
            left_alpha = right_alpha

        # Refine using midpoint method
        mid_alpha = 0.5 * (left_alpha + right_alpha)
        this_pfa = compute_pfa(mid_alpha, N)

        while abs(this_pfa - Pfa_set) > 0.000001 * Pfa_set:
            if this_pfa > Pfa_set:
                left_alpha = mid_alpha
            else:
                right_alpha = mid_alpha

            mid_alpha = 0.5 * (left_alpha + right_alpha)
            this_pfa = compute_pfa(mid_alpha, N)

        return mid_alpha

    position = []
    left_num = guard_num + ref_num
    start_cell = left_num + 1
    stop_cell = len(signal) - left_num
    N = 2 * ref_num
    alpha = get_alpha(Pfa, N)

    threshold = np.zeros(stop_cell - start_cell + 1)

    for ii in range(start_cell, stop_cell + 1):
        # Get left and right reference cells separately
        left_ref = signal[ii - left_num - 1:ii - guard_num - 1]  # Python indexing
        right_ref = signal[ii + guard_num:ii + left_num]  # Python indexing

        # Take the minimum of the two reference window averages
        tmp_data = min(np.mean(left_ref), np.mean(right_ref))
        tmp = tmp_data * alpha

        threshold_idx = ii - start_cell
        threshold[threshold_idx] = tmp

        if tmp < signal[ii - 1]:  # -1 for Python indexing
            position.append(ii)

    return np.array(position), threshold, start_cell, stop_cell

def generateDataGaussianWhite(num_cells, pos_target, echo_power_dB, noise_power_dB):
    """
    Generate signal data with Gaussian white noise and target echoes

    Parameters:
    -----------
    num_cells : int
        The amount of cells
    pos_target : array_like
        Positions of all targets
    echo_power_dB : float or array_like
        Echo wave power in dB (if array, should match length of pos_target)
    noise_power_dB : float
        Noise power in dB

    Returns:
    --------
    signal : ndarray
        Mixed data of signal and noise
    """

    num_target = len(pos_target)

    # Convert dB to linear power
    echo_power = 10 ** (np.array(echo_power_dB) / 10)
    noise_power = 10 ** (noise_power_dB / 10)

    # Generate complex Gaussian noise
    noise_sample = np.random.randn(num_cells) + 1j * np.random.randn(num_cells)
    # Calculate noise magnitude squared (power)
    noise = np.abs(np.sqrt(noise_power / 2) * noise_sample) ** 2

    if num_target == 0:
        signal = noise
    else:
        signal_sample = np.zeros(num_cells, dtype=complex)

        # Ensure echo_power is an array of the same length as pos_target
        if np.isscalar(echo_power):
            echo_power = np.full(num_target, echo_power)

        for ii in range(num_target):
            # Python uses 0-based indexing, so subtract 1 from target position
            signal_sample[pos_target[ii] - 1] = echo_power[ii]

        signal = signal_sample + noise

    return signal

































