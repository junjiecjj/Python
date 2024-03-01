#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:27:32 2024

@author: jack
"""

"""
滤波模块
"""

import numpy as np
import scipy.signal as signalP


def butter_lowpass(cutoff, fs, order=5):
    """
    根据低通滤波通带截至频率和采样频率计算滤波器分子系数b和分母系数a
    :param cutoff: 截至频率
    :param fs:     采样频率
    :param order:  滤波器的阶数
    :return: b, a分别为滤波器的分子和分母
    """
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signalP.butter(order, normal_cutoff, btype='low')
    return b, a


def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    """
    对信号作低通滤波
    :param data:     输入信号
    :param cutoff:  通带截至频率
    :param fs:     采样频率
    :param order:  滤波器的阶数
    :return:       返回值为y,经低通滤波后的信号
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    shape = data.shape
    if shape[0] != 1:
        data = data.T
    y = signalP.filtfilt(b, a, data)
    y =y.T
    return y
