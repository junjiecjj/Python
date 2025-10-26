#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:56:38 2025

@author: jack
"""

import numpy as np
from scipy.special import comb

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





