#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 00:10:35 2025

@author: jack
"""
import scipy
import numpy as np
# import statsmodels.tsa.api as smt
# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
import commpy



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
    def __init__(self, x = None, B = 4, maxerror = 1e-2, maxIter = 200, funtype = 'gauss'):
        """
        Parameters
        ----------
        x : init data for determinating of quantization threshold and represent point.
        B : quantization bit width.
        maxerror : max error to stop iteration.
        funtype: the distribution of data wanted to be quantized. 'gauss' or 'laplace', or any given fun.
        Returns: None.
        """
        self.B = B
        self.M = int(2**B)
        if funtype == "gauss":
            self.f = gauss
            f = gauss
            expected_f = expected_gauss
        elif funtype == "laplace":
            self.f = laplace
            f = laplace
            expected_f = expected_laplace
        self.thres = None
        self.represent = None
        if x == None:
            x = np.random.randn(20000)
        errors = []
        represent = self.initRepresent(self.M, x)
        thres = self.updateThreshold(represent)
        error = self.MSE(thres, represent, f)
        errors.append(error)
        Iter = 0
        while error > maxerror and Iter < maxIter:
            Iter += 1
            represent = self.updateRepresent(thres, expected_f, f)
            thres = self.updateThreshold(represent)
            error = self.MSE(thres, represent, f)
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

    def quantize_float(self, x ):
        """Quantization operation.
        """
        thre = np.append(self.thres, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        x_hat_q = np.zeros(np.shape(x))
        for i in range(thre.size - 1):
            x_hat_q = np.where(np.logical_and(x > thre[i], x <= thre[i+1]), np.full(np.size(x_hat_q), self.represent[i]), x_hat_q)

        return x_hat_q

    def quantize_bits(self, x, ):
        thre = np.append(self.thres, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        x_hat = np.zeros(np.shape(x), dtype= np.int32)
        for i in range(thre.size - 1):
            x_hat = np.where(np.logical_and(x > thre[i], x <= thre[i+1]), np.full(np.size(x_hat), int(i)), x_hat)

        bits = np.zeros((x_hat.size * self.B, ), dtype = np.int8)

        for idx, num in enumerate(x_hat):
            bits[idx*self.B : (idx+1)*self.B] = [int(b) for b in np.binary_repr(num, width = self.B)]
        return bits

    def dequantize_bits(self, bits, ):
        mapfunc = np.vectorize(lambda i:  commpy.utilities.bitarray2dec(bits[i:i + self.B ]))
        idx = mapfunc(np.arange(0, len(bits), self.B))
        quantized = self.represent[idx]
        return quantized

    def plot(self, ):
        import matplotlib.patches as mpatches

        thre = np.append(self.thres, self.thres[-1] + 1)
        thre = np.insert(thre, 0, self.thres[0] - 1)
        ##### plot
        fig, axs = plt.subplots(1, 1, figsize = (16, 6), constrained_layout = True)
        x = np.arange(-6, 6, 0.01)
        y = self.f(x)
        axs.plot(x, y, ls = '-', c = 'b', )

        axs.scatter(self.thres, np.zeros(self.thres.size), marker = '|', s = 200, c = 'k', )
        axs.scatter(self.represent, np.zeros(self.represent.size), marker = 'o', s = 60, c = 'r', )

        for x in self.thres:
            axs.vlines(x, ymin = 0, ymax = y.max()/2 + 0.1, colors = 'gray', ls = '--')

        for i, rep in enumerate(self.represent):
            arr = mpatches.FancyArrowPatch((thre[i], y.max()/2), (thre[i+1], y.max()/2), arrowstyle='<|-|>, head_length=0.4, head_width=0.15', mutation_scale=20, color = 'r')
            axs.add_patch(arr)
            axs.annotate(f"R$_{i}$:{np.binary_repr(i, width = self.B)}", xy = (.5, .5), xycoords=arr, horizontalalignment='center', verticalalignment='bottom', fontsize = 16, color = 'r')
            axs.text(self.represent[i], 0.02, f"{self.represent[i]:.2f}",  horizontalalignment="center", verticalalignment="center", fontsize = 16)
        axs.set_xticks(self.thres)

        axs.spines['bottom'].set_linewidth(2) ###设置底部坐标轴的粗细
        axs.set_xlim(self.thres[0] - 2, self.thres[-1] + 2)
        axs.set_ylim(-0.1 , y.max() + 0.1 )  #拉开坐标轴范围显示投影

        # axs.set_xticks([])
        axs.set_yticks([])

        axs.spines['bottom'].set_position(('data', 0))
        for i in ['top', 'right', 'left']: # 不显示刻度轴
            axs.spines[i].set_visible(False)
        axs.tick_params(labelsize = 16, top = False, left = False, right = False) # 不显示刻度

        axs.set_title(f"{self.B}-bit LloydMax quantizer", fontsize = 28 )

        plt.show()
        plt.close()

        fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
        axs.plot(self.errors, ls = '-', c = 'b', )

        axs.tick_params(direction = 'in', axis = 'both', top = True, right = True, labelsize = 25, width=3,)
        labels = axs.get_xticklabels() + axs.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # [label.set_fontsize(24) for label in labels]  # 刻度值字号

        # axs.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
        # axs.set_ylim(0.5, 1.0)  #拉开坐标轴范围显示投影

        axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
        axs.spines['bottom'].set_linewidth(2)    ### 设置底部坐标轴的粗细
        axs.spines['left'].set_linewidth(2)      #### 设置左边坐标轴的粗细
        axs.spines['right'].set_linewidth(2)     ### 设置右边坐标轴的粗细
        axs.spines['top'].set_linewidth(2)       #### 设置上部坐标轴的粗细

        font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 28}
        axs.set_xlabel( "Iter round", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
        axs.set_ylabel('MSE', fontproperties=font2, )
        plt.show()
        plt.close()
        return

q = LloydMaxQuantizer(B = 3, funtype = 'laplace')
print(f"represent = \n{q.represent}\nthres = \n{q.thres}")
q.plot()



















































































































































































