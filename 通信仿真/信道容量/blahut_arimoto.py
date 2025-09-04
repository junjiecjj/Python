#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:20:18 2025

@author: jack

https://github.com/kobybibas/blahut_arimoto_algorithm/blob/master/blahut_arimoto_algorithm.py

https://blog.csdn.net/mrcharles/article/details/92759378

https://www.cnblogs.com/HuangYJ/p/11826777.html

"""

import numpy as np
def blahut_arimoto(p_y_x: np.ndarray, log_base: float = 2, thresh: float = 1e-12, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    p_y_x: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''
    ## Input test
    assert np.abs(p_y_x.sum(axis=1).mean() - 1) < 1e-6
    assert p_y_x.shape[0] > 1
    ## The number of inputs: size of |X|
    m = p_y_x.shape[0]
    ## The number of outputs: size of |Y|
    n = p_y_x.shape[1]
    ## Initialize the prior uniformly
    px = np.ones((1, m)) / m
    ## Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):
        q_x_y = px.T * p_y_x
        q_x_y = q_x_y / np.sum(q_x_y, axis=0)

        px1 = np.prod(np.power(q_x_y, p_y_x), axis=1)
        px1 = px1 / np.sum(px1)

        tolerance = np.linalg.norm(px1 - px)
        px = px1
        if tolerance < thresh:
            break
    ## Calculate the capacity
    px = px.flatten()
    c = 0
    for i in range(m):
        if px[i] > 0:
            c += np.sum(px[i] * p_y_x[i, :] * np.log(q_x_y[i, :] / px[i] + 1e-16))
    c = c / np.log(log_base)
    return c, px

# # Example
# ## Binary symmetric channel
# The BSC is a binary channel; that is, it can transmit only one of two symbols (usually called 0 and 1). <br>
# The transmission is not perfect, and occasionally the receiver gets the wrong bit.  <br>
# The capacity of this channel  <br>
# $C = 1 - H_b(P_e)$

# In[76]:
e = 0.2
p1 = [1-e, e]
p2 = [e, 1-e]
p_y_x = np.asarray([p1, p2])
C, r = blahut_arimoto(p_y_x)
print('Capacity: ', C)
print('The prior: ', r)

# The analytic solution of the capaciy
H_P_e = - e * np.log2(e) - (1-e) * np.log2(1-e)
print('Anatliyic capacity: ', (1 - H_P_e))


# ## Erasure channel
# A binary erasure channel (or BEC) is a common communications channel.  <br>
# In this model, a transmitter sends a bit (a zero or a one), and the receiver either receives the bit or it receives a message that the bit was not received ("erased").  <br>
# The capacity of this channel is  <br>
# $C = 1 - P_e$.

# In[77]:

e = 0.1
p1 = [1-e, e, 0]
p2 = [0, e, 1-e]
p_y_x = np.asarray([p1, p2])
C, r = blahut_arimoto(p_y_x, log_base=2)
print('Capacity: ', C)
print('The prior: ', r)

# The analytic solution of the capaciy
print('Anatliyic capacity: ', (1 - e))
