#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:24:14 2024

@author: jack
"""



# This notebook illustrates how to use the libraries and functions in the
# sparc_public repository to run SPARC encoding + AMP decoding simulations
# and state evolution simulations.
#
# Copyright (c) 2021 Kuan Hsieh

import numpy as np
from sparc_sim import sparc_sim
from sparc_se import sparc_se
import matplotlib.pyplot as plt
import time



# 1. Regular SPARCs¶
# SPARC with AMP decoding simulations

awgn_var      = 1.0            # AWGN channel noise variance
code_params   = {'P': 15.0,    # Average codeword symbol power constraint
                 'R': 1.3,     # Rate
                 'L': 1000,    # Number of sections
                 'M': 32}      # Columns per section
decode_params = {'t_max': 25}  # Maximum number of iterations
num_of_runs   = 10             # Number of encoding/decoding trials
rng = np.random.RandomState(seed=None) # Random number generator

nmse_store  = np.zeros((num_of_runs, decode_params['t_max']))
for i in range(num_of_runs):
    start_time    = time.perf_counter()
    rng_seed      = rng.randint(2**32-1, size=2).tolist()
    results       = sparc_sim(code_params, decode_params, awgn_var, rng_seed)
    nmse_store[i] = results['nmse']
    print('Run #{}, SER: {:1.4f}, number of iterations: {:3d}, time elapsed: {:2.3f}'
          .format(i, results['ser'], results['t_final'], time.perf_counter()-start_time))
print('Code parameters:', code_params)

# State evolution
mc_samples = 1000000
nmse_se,_ = sparc_se(awgn_var, code_params, decode_params['t_max'], mc_samples)

plt.figure(figsize=(15,4))
plt.subplot(121)
_=plt.plot(nmse_store.T, linewidth=1.0, color='C0')
plt.xlabel('iteration')
plt.ylabel('NMSE')
plt.subplot(122)
_=plt.plot(nmse_store.mean(axis=0), linewidth=1.0, color='C1')
_=plt.plot(nmse_se, linewidth=1.0, color='C2')
plt.legend(['Average over {} runs'.format(num_of_runs), 'SE prediction'])
plt.xlabel('iteration')
plt.ylabel('NMSE')


# 2. Power allocated SPARCs
# SPARC with AMP decoding simulations
awgn_var      = 1.0                 # AWGN channel noise variance
code_params   = {'power_allocated': True,
                 'P': 15.0,         # Average codeword symbol power constraint
                 'R': 1.4,          # Rate
                 'L': 1024,         # Number of sections
                 'M': 32,           # Columns per section
                 'B': 32,           # Number of different 'powers'
                 'R_PA_ratio': 0.9} # Parameter for iterative power allocation
decode_params = {'t_max': 25}       # Maximum number of iterations
num_of_runs   = 10                  # Number of encoding/decoding trials
rng = np.random.RandomState(seed=None)

nmse_store  = np.zeros((num_of_runs, decode_params['t_max']))
for i in range(num_of_runs):
    start_time    = time.perf_counter()
    rng_seed      = rng.randint(2**32-1, size=2).tolist()
    results       = sparc_sim(code_params, decode_params, awgn_var, rng_seed)
    nmse_store[i] = results['nmse'].mean(axis=1)
    print('Run #{}, SER: {:1.4f}, number of iterations: {:3d}, time elapsed: {:2.3f}'
          .format(i, results['ser'], results['t_final'], time.perf_counter()-start_time))
print('Code parameters:', code_params)


# State evolution
mc_samples = 100000
nmse_se,_ = sparc_se(awgn_var, code_params, decode_params['t_max'], mc_samples)

plt.figure(figsize=(15,4))
plt.subplot(121)
_=plt.plot(nmse_store.T, linewidth=1.0, color='C0')
plt.xlabel('iteration')
plt.ylabel('NMSE')
plt.subplot(122)
_=plt.plot(nmse_store.mean(axis=0), linewidth=1.0, color='C1')
_=plt.plot(nmse_se.mean(axis=1), linewidth=1.0, color='C2')
plt.legend(['Average over {} runs'.format(num_of_runs), 'SE prediction'])
plt.xlabel('iteration')
plt.ylabel('NMSE')

# 3. Spatially coupled SPARCs¶

# SPARC with AMP decoding simulations

awgn_var      = 1.0                 # AWGN channel noise variance
code_params   = {'spatially_coupled': True,
                 'P': 15.0,         # Average codeword symbol power constraint
                 'R': 1.4,          # Rate
                 'L': 1024,         # Number of sections
                 'M': 32,           # Columns per section
                 'omega': 2,        # Coupling width
                 'Lambda': 8}       # Coupling length
decode_params = {'t_max': 25}       # Maximum number of iterations
num_of_runs   = 10                  # Number of encoding/decoding trials
rng = np.random.RandomState(seed=None)

nmse_store  = np.zeros((num_of_runs, decode_params['t_max']))
for i in range(num_of_runs):
    start_time    = time.perf_counter()
    rng_seed      = rng.randint(2**32-1, size=2).tolist()
    results       = sparc_sim(code_params, decode_params, awgn_var, rng_seed)
    nmse_store[i] = results['nmse'].mean(axis=1)
    print('Run #{}, SER: {:1.4f}, number of iterations: {:3d}, time elapsed: {:2.3f}'
          .format(i, results['ser'], results['t_final'], time.perf_counter()-start_time))
print('Code parameters:', code_params)

# State evolution
mc_samples = 100000
nmse_se,_ = sparc_se(awgn_var, code_params, decode_params['t_max'], mc_samples)

plt.figure(figsize=(15,4))
plt.subplot(121)
_=plt.plot(nmse_store.T, linewidth=1.0, color='C0')
plt.xlabel('iteration')
plt.ylabel('NMSE')
plt.subplot(122)
_=plt.plot(nmse_store.mean(axis=0), linewidth=1.0, color='C1')
_=plt.plot(nmse_se.mean(axis=1), linewidth=1.0, color='C2')
plt.legend(['Average over {} runs'.format(num_of_runs), 'SE prediction'])
plt.xlabel('iteration')
plt.ylabel('NMSE')


# 4. Modulated complex SPARCs
# SPARC with AMP decoding simulations

awgn_var      = 1.0                 # AWGN channel noise variance
code_params   = {'complex':   True,
                 'modulated': True,
                 'P': 15.0,         # Average codeword symbol power constraint
                 'R': 1.3*2,        # Rate
                 'L': 1024*2,       # Number of sections
                 'M': 8,            # Columns per section
                 'K': 4}            # Modulation factor
decode_params = {'t_max': 25}       # Maximum number of iterations
num_of_runs   = 10                  # Number of encoding/decoding trials
rng = np.random.RandomState(seed=None)

nmse_store  = np.zeros((num_of_runs, decode_params['t_max']))
for i in range(num_of_runs):
    start_time    = time.perf_counter()
    rng_seed      = rng.randint(2**32-1, size=2).tolist()
    results       = sparc_sim(code_params, decode_params, awgn_var, rng_seed)
    nmse_store[i] = results['nmse']
    print('Run #{}, SER: {:1.4f}, number of iterations: {:3d}, time elapsed: {:2.3f}'
          .format(i, results['ser'], results['t_final'], time.perf_counter()-start_time))
print('Code parameters:', code_params)
# State evolution
mc_samples = 1000000
nmse_se,_ = sparc_se(awgn_var, code_params, decode_params['t_max'], mc_samples)

plt.figure(figsize=(15,4))
plt.subplot(121)
_=plt.plot(nmse_store.T, linewidth=1.0, color='C0')
plt.xlabel('iteration')
plt.ylabel('NMSE')
plt.subplot(122)
_=plt.plot(nmse_store.mean(axis=0), linewidth=1.0, color='C1')
_=plt.plot(nmse_se, linewidth=1.0, color='C2')
plt.legend(['Average over {} runs'.format(num_of_runs), 'SE prediction'])
plt.xlabel('iteration')
plt.ylabel('NMSE')



























































































































