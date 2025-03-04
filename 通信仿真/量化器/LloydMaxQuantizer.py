#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 00:10:35 2025

@author: jack
"""
import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# import commpy



def gauss(x, mean = 0.0, vari = 1.0):
    return (1.0/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean), 2.0))/(2.0*vari))
def expected_gauss(x, mean = 0.0, vari = 1.0):
    """
        A expected value of normal distribution function which created to use with scipy.integral.quad
    """
    return (x/(np.sqrt(2.0*np.pi*vari)))*np.exp((-np.power((x-mean), 2.0))/(2.0*vari))

def laplace(x, mean = 0.0, vari = 1.0):
    """
        A laplace distribution function to use with scipy.integral.quad
    """
    #In laplace distribution beta is used instead of variance so, the converting is necessary.
    scale = np.sqrt(vari/2.0)
    return (1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))
def expected_laplace(x, mean = 0.0, vari = 1.0):
    """
        A expected value of laplace distribution function which created to use with scipy.integral.quad
    """
    scale = np.sqrt(vari/2.0)
    return (x * 1.0/(2.0*scale))*np.exp(-(np.abs(x-mean))/(scale))


class LloydMaxQuantizer(object):
    """
        A class for iterative Lloyd Max quantizer. This quantizer is created to minimize amount SNR between the orginal signal and quantized signal.
    """
    def __init__(self, x = None, B = 4, maxerror = 1e-2, maxIter = 2000, funtype = 'gauss'):
        """
        Parameters
        ----------
        x : init data for determinating of quantization threshold and represent point.
        B : quantization bit width.
        maxerror : max error to stop iteration.
        funtype: the distribution of data wanted to be quantized. 'gauss' or 'laplace', or any given fun.
        Returns: None.
        """
        self.M = int(2**B)
        self.funtype = funtype

        self.thres = None
        self.represent = None
        if x == None:
            x = np.random.randn(20000)
        errors = []
        represent = self.initRepresent(self.M, x)
        thres = self.updateThreshold(represent)
        error = self.MSE(thres, represent, gauss)
        errors.append(error)
        Iter = 0

        while error > maxerror and Iter < maxIter:
            Iter += 1
            represent = self.updateRepresent(thres, expected_gauss, gauss)
            thres = self.updateThreshold(represent)
            error = self.MSE(thres, represent, gauss)
            errors.append(error)
        self.thres = thres
        self.represent = represent
        self.errors = errors
        return

    @staticmethod
    def initRepresent(M, x):
        """ Use the data x to init the representation
        """
        x = np.array(x).flatten()
        # num_repre  = np.power(2, bit)
        step = (np.max(x) - np.min(x))/M

        middle_point = np.mean(x)
        repre = np.array([])
        for i in range(int(M/2)):
             repre = np.append(repre, middle_point + (i+1)*step)
             repre = np.insert(repre, 0, middle_point - (i+1)*step)
        assert repre.size == M
        return repre

    @staticmethod
    def updateThreshold(repre):
        """ Update the threshold according to the current representation.
        """
        t_q = np.zeros(repre.size - 1)
        for i in range(t_q.size):
            t_q[i] = (repre[i] + repre[i+1])/2.0
        return t_q

    @staticmethod
    def updateRepresent(thre, expected_dist, dist):
        """ Update the representation according to the current threshold.
        """
        repres = np.zeros(thre.size + 1)
        thre = np.array(thre).copy()
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)

        for i in range(repres.size):
             repres[i] = scipy.integrate.quad(expected_dist, thre[i], thre[i+1])[0]/(scipy.integrate.quad(dist, thre[i], thre[i+1])[0])
        return repres

    @staticmethod
    def MSE(thre, repre, f):
        assert repre.size == thre.size + 1
        MSE = 0.0
        thre = np.array(thre)
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        for i in range(repre.size):
            tmp = scipy.integrate.quad(lambda t: ((t - repre[i])**2) * f(t), thre[i], thre[i+1])[0]
            MSE += tmp
        return MSE

    @staticmethod
    def quantize_float(x, thre, repre):
        """Quantization operation.
        """
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        x_hat_q = np.zeros(np.shape(x))
        for i in range(len(thre)-1):
            if i == 0:
                x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]), np.full(np.size(x_hat_q), repre[i]), x_hat_q)
            elif i == range(len(thre))[-1]-1:
                x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]), np.full(np.size(x_hat_q), repre[i]), x_hat_q)
            else:
                x_hat_q = np.where(np.logical_and(x > thre[i], x < thre[i+1]), np.full(np.size(x_hat_q), repre[i]), x_hat_q)
        return x_hat_q

    @staticmethod
    def quantize_bits(x, thres, represent):

        return

    @staticmethod
    def dequantize_bits(x, thres, represent):

        return


q = LloydMaxQuantizer(B = 2)





























































































































































































