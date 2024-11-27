# Python code to run State Evolution (SE) related functions for
# Sparse Regression Codes (SPARCs)
#
# 1. State evolution for general base matrix W.
# 2. Asymptotic state evolution for general base matrix W.
#
# Copyright (c) 2020 Kuan Hsieh

import numpy as np
from sparc import create_base_matrix, is_power_of_2, psk_constel
from copy import copy

### Check code/decode/channel params functions
def check_code_params(code_params):
    '''
    Check SPARC code parameters for State Evolution (SE) simulations
    '''

    code_params_copy = {} # Will overwrite original (prevents unwanted params)

    def in_code_param_list(code_param_list):
        if all([(item in code_params) for item in code_param_list]):
            for item in code_param_list:
                code_params_copy[item] = copy(code_params[item])
        else:
            raise Exception('Need code parameters {}.'.format(code_param_list))

    # Check SPARC type e.g. power allocated, spatially coupled
    sparc_type_list = ['complex',
                       'modulated',
                       'power_allocated',
                       'spatially_coupled']
    for item in sparc_type_list:
        if item not in code_params:
            code_params[item] = False # default
        else:
            assert type(code_params[item]) == bool,\
                    "'{}' must be boolean".format(key)
        code_params_copy[item] = copy(code_params[item])

    # Required SPARC code parameters (all SPARC types)
    code_param_list = ['P','R','M']
    in_code_param_list(code_param_list)
    P,R,M = map(code_params.get, code_param_list)
    assert (type(P)==float or type(P)==np.float64) and P>0
    assert (type(R)==float or type(R)==np.float64) and R>0
    assert type(M)==int and M>0 and is_power_of_2(M)

    # Required SPARC code parameters (modulated)
    # ONLY SUPPORTS PSK MODULATION
    if code_params['modulated']:
        code_param_list = ['K']
        in_code_param_list(code_param_list)
        K = code_params['K']
        assert type(K)==int and K>1 and is_power_of_2(K)
        if not code_params['complex']:
            assert K==2, 'Real-modulated SPARCs requires K=2'

    # Required SPARC code parameters (power allocated)
    # ONLY SUPPORTS ITERATIVE POWER ALLOCATION
    if code_params['power_allocated']:
        code_param_list = ['B', 'R_PA_ratio']
        in_code_param_list(code_param_list)
        B, R_PA_ratio = map(code_params.get, code_param_list)
        assert type(B)==int and B>1
        assert type(R_PA_ratio)==float or type(R_PA_ratio)==np.float64
        assert R_PA_ratio>=0

    # Required SPARC code parameters (spatially coupled)
    # ONLY SUPPORTS OMEGA, LAMBDA BASE MATRICES
    if code_params['spatially_coupled']:
        code_param_list = ['omega', 'Lambda']
        in_code_param_list(code_param_list)
        omega, Lambda = map(code_params.get, code_param_list)
        assert type(omega)==int and omega>1
        assert type(Lambda)==int and Lambda>=(2*omega-1)

    # Overwrite orignal
    code_params.clear()
    code_params.update(dict(code_params_copy))

def sparc_se_E(tau, K, u):

    itau  = 1/tau
    rtau  = np.sqrt(itau)

    if K == 1:
        expsA = np.exp(itau + rtau * u[:,0])
        expsB = expsA
        expsC = np.exp(rtau * u[:,1:])
    elif K == 2:
        expsA = np.sinh(itau + rtau * u[:,0])
        expsB = expsA
        expsC = np.cosh(rtau * u[:,1:])
    elif K == 4: # Must be complex
        expsA = np.sinh(itau + rtau * u[:,0].real)
        expsB = np.cosh(itau + rtau * u[:,0].real) + np.cosh(rtau * u[:,0].imag)
        expsC = np.cosh(rtau * u[:,1:].real) + np.cosh(rtau * u[:,1:].imag)
    else:        # Must be complex
        n, M = u.shape
        c    = psk_constel(K)
        tmpA = np.zeros((K, n))
        tmpB = np.zeros((K, n))
        tmpC = np.zeros((K, n, M-1))
        for k in range(K):
            tmpB[k] = np.exp(np.real((itau + rtau * u[:,0]) * c[k].conj()))
            tmpA[k] = np.real(c[k]) * tmpB[k]
            tmpC[k] = np.exp(np.real((rtau * u[:,1:]) * c[k].conj()))
        expsA = tmpA.mean(axis=0)
        expsB = tmpB.mean(axis=0)
        expsC = tmpC.mean(axis=0)

    E = expsA / (expsB + expsC.sum(axis=1))

    return E.mean()

def sparc_se(awgn_var, code_params, t_max, mc_samples):
    """
    State evolution (SE) for Sparse Regression Codes.

    I resuse the Monto Carlo samples instead of resampling them
    at everytime to save computation.

    awgn_var   : AWGN channel noise variance
    code_params: SPARC code parameters
    t_max      : max number of iterations
    mc_samples : num of Monte Carlo samples
    """

    check_code_params(code_params)

    # Construct base matrix W
    tmp = code_params.copy()
    tmp.update({'awgn_var':awgn_var})
    W   = create_base_matrix(**tmp)
    assert 0 <= W.ndim <= 2

    # Get code parameters
    P,R,M = map(code_params.get, ['P','R','M'])
    K = code_params['K'] if code_params['modulated'] else 1

    if code_params['complex']:
        R /= 2 # Complex SPARCs only care about the rate per dimension

    if W.ndim == 0:
        psi = np.ones(t_max)
    elif W.ndim == 1:
        Lr, Lc = 1, W.size
        psi = np.ones((t_max, Lc))
    elif W.ndim == 2:
        Lr, Lc = W.shape
        psi = np.ones((t_max, Lc))

    if K>2: # Must be complex and modulated with modulation factor K>2
        u = np.random.randn(mc_samples, M) + 1j*np.random.randn(mc_samples, M)
    else:
        u = np.random.randn(mc_samples, M)

    for t in range(t_max-1):
        if t>0:
            tau_prev = np.copy(tau)

        if W.ndim == 0:
            tau = (np.log(2)*R/np.log(K*M)) * (awgn_var/P + psi[t])
        else:
            phi = awgn_var + np.dot(W, psi[t])/Lc
            tau = (np.log(2)*R*Lr/np.log(K*M)) / np.dot(W.T, 1/phi)

        if (t>0) and np.allclose(tau, tau_prev, rtol=1e-6, atol=0):
            if W.ndim == 0:
                psi[t:] = psi[t]
            else:
                psi[t:,:] = psi[t,:]
            break

        if W.ndim == 0:
            psi[t+1] = 1 - sparc_se_E(tau, K, u)
        else:
            for c in range(Lc):
                psi[t+1, c] = 1 - sparc_se_E(tau[c], K, u)

    # Final tau can be used to estimate SER
    return psi, tau

