#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:54:47 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODY4ODE5MQ==&mid=2247485008&idx=1&sn=aff1ee8900712b2e000dc734a7454ab8&chksm=cef1dcf8350c01cee28b982ad23f0e1fffff3610642d2bb47bae011cb7d61a5e88fa3135bf34&mpshare=1&scene=1&srcid=0901avGpb5aT3RYj9I8lh6bz&sharer_shareinfo=918f55b22596bee30c43a7d462e9c4d6&sharer_shareinfo_first=918f55b22596bee30c43a7d462e9c4d6&exportkey=n_ChQIAhIQFJaES3mS2SG8jEnPQslU%2BhKfAgIE97dBBAEAAAAAAIg%2BFRLHqgMAAAAOpnltbLcz9gKNyK89dVj0SzXeR0YxqcVwFu2UfoaKxWTYe1XcuIIjdJnGdjrjHwAQRHeFyTFTAuTBfWQzV%2FLUQcv4qsc%2FfDiXMMo%2BXAYJG09XP68j9QYmTWJEDMYof3tbmd887I3OnoYfhc1AxvJs4laq93zeUVASf93MHPs7%2FlH78hIIshANOh1Bsvae%2F5wbGuaUyN%2BHLqkztGsCWgRlevJ2mMOoY6hvioXOJ9CP%2Fo8%2Fd1VNHmYU3mYXFLiMMw1%2F%2BPSbkhCIj5mrH%2BfUY3QnDZFCOKoiqEx0glSM9zVJKnaaX2ydJguQ2bCpi5D%2FunS7M3gBIBVZ6Nu6a2JoKhGaCcUbAj%2BHBAL0&acctmode=0&pass_ticket=CXObqEV2Wc06cx5rDcJsTQw4e7pPRj%2F1uub6BK%2BNi33hnV5LHlPB2Phk%2FjQnxy2I&wx_header=0#rd


"""



from numpy import asarray, exp
from numpy.random import randn, rand, seed
from matplotlib import pyplot

# Define objective function
def objective(step):
    return step[0] ** 2.0

# Define simulated annealing algorithm
def sa(objective, area, iterations, step_size, temperature):
    # create initial point
    start_point = area[:, 0] + rand( len( area ) ) * ( area[:, 1] - area[:, 0] )
    # evaluate initial point
    start_point_eval = objective(start_point)
    # Assign previous and new solution to previous and new_point_eval variable
    mia_start_point, mia_start_eval = start_point, start_point_eval
    outputs = []
    for i in range(iterations):
        # First step by mia
        mia_step = mia_start_point + randn( len( area ) ) * step_size
        mia_step_eval = objective(mia_step)
        if mia_step_eval < start_point_eval:
            start_point, start_point_eval = mia_step, mia_step_eval
      #Append the new values into the output list
            outputs.append(start_point_eval)
            print('Acceptance Criteria = %.5f' % mac," ",'iteration Number = ',i," ", 'best_so_far = ',start_point," " ,'new_best = %.5f' % start_point_eval)
        difference = mia_step_eval - mia_start_eval
        t = temperature / float(i + 1)
        # calculate Metropolis Acceptance Criterion / Acceptance Probability
        mac = exp(-difference / t)
        # check whether the new point is acceptable
        if difference < 0 or rand() < mac:
            mia_start_point, mia_start_eval = mia_step, mia_step_eval
    return [start_point, start_point_eval, outputs]

seed(1)
# define the area of the search space
area = asarray([[-6.0, 6.0]])
# initial temperature
temperature = 12
# define the total no. of iterations
iterations = 1200
# define maximum step_size
step_size = 0.1
# perform the simulated annealing search
start_point, output, outputs = sa(objective, area, iterations, step_size, temperature)
#plotting the values
pyplot.plot(outputs, 'ro-')
pyplot.xlabel('Improvement Value')
pyplot.ylabel('Evaluation of Objective Function')
pyplot.show()
























