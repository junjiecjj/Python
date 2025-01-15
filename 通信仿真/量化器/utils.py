#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:01:09 2025

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collection of utility functions for Lloyd Max Quantizer.
@author: Ninnart Fuengfusin
"""
import numpy as np
import scipy.integrate as integrate


def normal_dist(x, mean=0.0, vari=1.0):
    """A normal distribution function created to use with scipy.integral.quad
    """
    return (1.0/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0))/(2.0*vari))


def expected_normal_dist(x, mean=0.0, vari=1.0):
    """A expected value of normal distribution function which created to use with scipy.integral.quad
    """
    return (x/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean),2.0))/(2.0*vari))


def laplace_dist(x, mean=0.0, vari=1.0):
    """ A laplace distribution function to use with scipy.integral.quad
    """
    #In laplace distribution beta is used instead of variance so, the converting is necessary.
    scale = np.sqrt(vari/2.0)
    return (1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))

def expected_laplace_dist(x, mean=0.0, vari=1.0):
    """A expected value of laplace distribution function which created to use with scipy.integral.quad
    """
    scale = np.sqrt(vari/2.0)
    return x*(1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))

#def variance(x, mean=0.0, std=1.0):
#    """
#    create normal distribution
#    """
#    return (1.0/(std*np.sqrt(2.0*np.pi)))*np.power(x-mean,2)*np.exp((-np.power((x-mean),2.0)/(2.0*np.power(std,2.0))))

def MSE_loss(x, x_hat_q):
    """Find the mean square loss between x (orginal signal) and x_hat (quantized signal)
    Args:
        x: the signal without quantization
        x_hat_q: the signal of x after quantization
    Return:
        MSE: mean square loss between x and x_hat_q
    """
    #protech in case of input as tuple and list for using with numpy operation
    x = np.array(x)
    x_hat_q = np.array(x_hat_q)
    assert np.size(x) == np.size(x_hat_q)
    MSE = np.sum(np.power(x-x_hat_q,2))/np.size(x)
    return MSE


class LloydMaxQuantizer(object):
    """A class for iterative Lloyd Max quantizer.
    This quantizer is created to minimize amount SNR between the orginal signal
    and quantized signal.
    """
    @staticmethod
    def start_repre(x, bit):
        """
        Generate representations of each threshold using
        Args:
            x: input signal for
            bit: amount of bit
        Return:
            threshold:
        """
        assert isinstance(bit, int)
        x = np.array(x)
        num_repre  = np.power(2,bit)
        step = (np.max(x)-np.min(x))/num_repre

        middle_point = np.mean(x)
        repre = np.array([])
        for i in range(int(num_repre/2)):
             repre = np.append(repre, middle_point+(i+1)*step)
             repre = np.insert(repre, 0, middle_point-(i+1)*step)
        return repre

    @staticmethod
    def threshold(repre):
        """
        """
        t_q = np.zeros(np.size(repre)-1)
        for i in range(len(repre)-1):
            t_q[i] = 0.5*(repre[i]+repre[i+1])
        return t_q

    @staticmethod
    def represent(thre, expected_dist, dist):
        """
        """
        thre = np.array(thre)
        x_hat_q = np.zeros(np.size(thre)+1)
        #prepare for all possible integration range
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)

        for i in range(len(thre)-1):
             x_hat_q[i] = integrate.quad(expected_dist, thre[i], thre[i+1])[0]/(integrate.quad(dist,thre[i],thre[i+1])[0])
        return x_hat_q

    @staticmethod
    def quant(x, thre, repre):
        """Quantization operation.
        """
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        x_hat_q = np.zeros(np.shape(x))
        for i in range(len(thre)-1):
            if i == 0:
                x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]),
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
            elif i == range(len(thre))[-1]-1:
                x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]),
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
            else:
                x_hat_q = np.where(np.logical_and(x > thre[i], x < thre[i+1]),
                                   np.full(np.size(x_hat_q), repre[i]), x_hat_q)
        return x_hat_q

if __name__ == '__main__':
    print('Please compile with main.py, this file is a collection of functions only.')
