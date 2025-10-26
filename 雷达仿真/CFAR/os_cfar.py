#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:41:58 2025

@author: jack
"""

import numpy as np
from scipy.special import comb, beta as beta_func

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




