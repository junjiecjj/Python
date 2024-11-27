# Python code to run Sparse Regression Code (SPARCs) simulations
#
# Copyright (c) 2020 Kuan Hsieh

import numpy as np
from scipy.fftpack import dct, idct
import time
from copy import copy
######## Miscellaneous ########

def is_power_of_2(x):
    return (x > 0) and ((x & (x - 1)) == 0)

########## Encoder/Decoder pairs ##########

### Main encode/decode functions
def sparc_encode(code_params, awgn_var, rand_seed):
    '''
    Encode message to SPARC codeword
    '''

    check_code_params(code_params)
    R,L,M = map(code_params.get,['R','L','M'])
    K = code_params['K'] if code_params['modulated'] else 1

    # Generate random bits
    bit_len = int(round(L*np.log2(K*M)))
    bits_in = rnd_bin_arr(bit_len, rand_seed)

    # Convert bits to message vector
    beta0 = bin_arr_2_msg_vector(bits_in, M, K)

    # Construct base matrix W
    tmp = code_params.copy()
    tmp.update({'awgn_var':awgn_var})
    W = create_base_matrix(**tmp)

    # Update code_params
    n = int(round(bit_len/R))   # Design codeword length
    if W.ndim == 2:
        Lr,_ = W.shape
        Mr   = int(round(n/Lr))
        n    = Mr * Lr          # Actual codeword length
    R_actual = bit_len / n      # Actual rate
    code_params.update({'n':n, 'R_actual':R_actual})

    # Set up function to calculate A*beta
    Ab, Az = sparc_transforms(W, L, M, n, rand_seed, code_params['complex'])

    # Generate random codeword
    x = Ab(beta0)

    return bits_in, beta0, x, Ab, Az

def sparc_decode(y, code_params, decode_params, awgn_var, rand_seed, beta0, Ab=None, Az=None):
    '''
    Decode SPARC codeword
    '''

    check_decode_params(decode_params)

    # Run AMP decoding (applies hard decision after final iteration)
    beta, t_final, nmse, psi = sparc_amp(y, code_params, decode_params,
                                         awgn_var, rand_seed, beta0, Ab, Az)

    # Whether or not we expect a section error.
    # Can be quite accurate for non power-allocated SPARCs with large M and L.
    expect_err = psi.mean() >= 0.001

    # Converts message vector back to bits
    K = code_params['K'] if code_params['modulated'] else 1
    bits_out = msg_vector_2_bin_arr(beta, code_params['M'], K)

    return bits_out, beta, t_final, nmse, expect_err

### Check code/decode/channel params functions
def check_code_params(code_params):
    '''
    Check SPARC code parameters
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
    code_param_list = ['P','R','L','M']
    in_code_param_list(code_param_list)
    P,R,L,M = map(code_params.get, code_param_list)
    assert (type(P)==float or type(P)==np.float64) and P>0
    assert (type(R)==float or type(R)==np.float64) and R>0
    assert type(L)==int and L>0
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
        assert L % B == 0, 'B must divide L'
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
        assert L % Lambda == 0, 'Lambda must divide L'

    if code_params['power_allocated'] and code_params['spatially_coupled']:
        assert L % (Lambda*B) == 0, 'Lambda*B must divide L'

    # Overwrite orignal
    code_params.clear()
    code_params.update(dict(code_params_copy))

def check_decode_params(decode_params):
    '''
    Check SPARC decode parameters
    '''
    # Required decode parameters
    decode_param_list = ['t_max']
    if not all([(item in decode_params) for item in decode_param_list]):
        raise Exception('Need decode parameters {}.'.format(decode_param_list))

    # Optional decode parameters
    decode_param_dic = {'rtol':1e-6, 'phi_est_method':1}
    for key, val in decode_param_dic.items():
        if key not in decode_params:
            decode_params[key] = val

    decode_param_list = ['t_max', 'rtol', 'phi_est_method']
    t_max, rtol, phi_est_method = map(decode_params.get, decode_param_list)
    assert type(t_max)==int and t_max>1
    assert type(rtol)==float and 0<rtol<1
    assert phi_est_method==1 or phi_est_method==2

######## Binary operations ########

def rnd_bin_arr(k, rand_seed):
    '''
    Generate random binary array (numpy.ndarray) of length k
    '''
    assert type(k) == int
    rng = np.random.RandomState(rand_seed)
    return rng.randint(2, size=k, dtype='bool')

def bin_arr_2_int(bin_array):
    '''
    Binary array (numpy.ndarray) to integer
    '''
    assert bin_array.dtype == 'bool'
    k = bin_array.size
    assert 0 < k < 64 # Ensures non-negative integer output
    return bin_array.dot(1 << np.arange(k)[::-1])

def int_2_bin_arr(integer, arr_length):
    '''
    Integer to binary array (numpy.ndarray) of length arr_length
    NB: only works for non-negative integers
    '''
    assert integer>=0
    return np.array(list(np.binary_repr(integer, arr_length))).astype('bool')
    # The folowing don't include array length
    #return np.array([int(x) for x in bin(integer)[2:]]).astype('bool')
    #return np.array(list(bin(integer)[2:])).astype('bool')
    #return np.array(list("{0:b}".format(integer))).astype('bool')

######## PSK modulation operations ########

# TAKEN FROM WIKIPEDIA
def bin2gray(num):
    '''
    Converts binary code (int type) to gray code (int type)
    From https://en.wikipedia.org/wiki/Gray_code
    '''
    return num ^ (num >> 1)

# TAKEN FROM WIKIPEDIA
def gray2bin(num):
    '''
    Converts gray code (int type) to binary code (int type)
    From https://en.wikipedia.org/wiki/Gray_code
    '''
    mask = num >> 1
    while (mask != 0):
        num  = num ^ mask
        mask = mask >> 1
    return num

def psk_constel(K):
    '''
    K-PSK constellation symbols
    '''
    assert type(K)==int and K>1 and is_power_of_2(K)

    if K == 2:
        c = np.array([1, -1])
    elif K == 4:
        c = np.array([1+0j, 0+1j, -1+0j, 0-1j])
    else:
        theta = 2*np.pi*np.arange(K)/K
        c     = np.cos(theta) + 1J*np.sin(theta)

    return c

def psk_mod(bin_arr, K):
    '''
    K-PSK modulation (using gray coding).

    bin_arr: boolean numpy.ndarray to modulate. Length of  L * log2(K).
    K      : number of PSK contellations, K>1 and is a power of 2

    Returns
    symbols: Corresponding K-PSK modulation symbols of length L.
             (If K=2 then symbols are real, complex otherwise.)
    '''

    assert type(K)==int and K>1 and is_power_of_2(K)
    assert bin_arr.dtype == 'bool'

    c    = psk_constel(K)           # Constellation symbols
    logK = int(round(np.log2(K)))
    assert bin_arr.size % logK == 0
    L    = bin_arr.size // logK     # Number of symbols
    if L == 1:
        idx     = gray2bin(bin_arr_2_int(bin_arr)) # gray code index
        symbols = c[idx]
    else:
        symbols = np.zeros(L, dtype=c.dtype)
        for l in range(L):
            idx        = gray2bin(bin_arr_2_int(bin_arr[l*logK:(l+1)*logK]))
            symbols[l] = c[idx]

    return symbols

def psk_demod(symbols, K):
    '''
    K-PSK demodulation (using gray coding).

    symbols: single symbol (float or complex) or np.ndarray of symbols.
    K      : number of PSK contellations, K>1 and is a power of 2

    Returns
    bin_arr: Corresponding boolean numpy.ndarray after demodulation.
             Has length L * log2(K) where L is the length of `symbols`.
    '''

    assert type(K)==int and K>1 and is_power_of_2(K)
    L    = symbols.size           # Number of symbols to demodulate
    c    = psk_constel(K)         # PSK constellation symbols
    logK = int(round(np.log2(K))) # Bits per symbol

    bin_arr = np.zeros(L*logK, dtype=bool)
    if L == 1:
        idx = bin2gray(np.argwhere(c == symbols))[0,0] # gray code index
        assert type(idx) == np.int64, 'Wrong type(idx)={}'.format(type(idx))
        bin_arr = int_2_bin_arr(idx, logK)
    else:
        for l in range(L):
            idx = bin2gray(np.argwhere(c == symbols[l]))[0,0]
            assert type(idx) == np.int64, 'Wrong type(idx)={}'.format(type(idx))
            bin_arr[l*logK:(l+1)*logK] = int_2_bin_arr(idx, logK)

    return bin_arr

######## Message vector operations ########

def rnd_msg_vector(L, M, rand_seed, K=1):
    '''
    Generate random message vector

    L: number of sections
    M: entries per section
    '''
    rng = np.random.RandomState(rand_seed)
    assert type(M)==int and M>0 and is_power_of_2(M)

    if K==1 or K==2:
        msg_vector = np.zeros((L,M))
    else:
        msg_vector = np.zeros((L,M), dtype=complex)

    idxs = rng.randint(0,M,L) # Locations of nonzeros

    if K==1:
        vals = np.ones(L)
    else:
        c    = psk_constel(K)
        vals = c[rng.randint(0,K,L)] # Values of nonzeros

    msg_vector[(np.arange(L), idxs)] = vals

    return msg_vector.ravel()

def bin_arr_2_msg_vector(bin_arr, M, K=1):
    '''
    Convert binary array (numpy.ndarray) to SPARC message vector

    M: entries per section of SPARC message vector
    K: parameter of K-PSK modulation for msg_vector (power of 2)
       If no modulation, K=1.
    '''
    assert type(M)==int and M>0 and is_power_of_2(M)
    logM = int(round(np.log2(M)))
    if K==1:
        sec_size = logM
    else:
        assert type(K)==int and K>1 and is_power_of_2(K)
        logK = int(round(np.log2(K)))
        sec_size = logM + logK

    bin_arr_size = bin_arr.size
    assert bin_arr_size % sec_size == 0
    L = bin_arr_size // sec_size # Num of sections

    if K==1 or K==2:
        msg_vector = np.zeros(L*M)
    else:
        msg_vector = np.zeros(L*M, dtype=complex)

    for l in range(L):
        idx = bin_arr_2_int(bin_arr[l*sec_size : l*sec_size+logM])
        if K==1:
            val = 1
        else:
            val = psk_mod(bin_arr[l*sec_size+logM : (l+1)*sec_size], K)
        msg_vector[l*M + idx] = val

    return msg_vector

def msg_vector_2_bin_arr(msg_vector, M, K=1):
    '''
    Convert SPARC message vector to binary array (numpy.ndarray)

    M: entries per section of SPARC message vector
    K: parameter of K-PSK modulation for msg_vector (power of 2)
       If no modulation, K=1.
    '''
    assert type(msg_vector) == np.ndarray
    assert type(M)==int and M>0 and is_power_of_2(M)
    assert msg_vector.size % M == 0
    logM = int(round(np.log2(M)))
    L = msg_vector.size // M

    if K==1:
        sec_size = logM
    else:
        assert type(K)==int and K>1 and is_power_of_2(K)
        logK = int(round(np.log2(K)))
        sec_size = logM + logK

    msg_reshape  = msg_vector.reshape(L,M)
    idxs1, idxs2 = np.nonzero(msg_reshape)
    assert np.array_equal(idxs1, np.arange(L)) # Exactly 1 nonzero in each row

    if K != 1:
        vals = msg_reshape[(idxs1, idxs2)] # Pick out the nonzero values

    bin_arr = np.zeros(L*sec_size, dtype='bool')
    for l in range(L):
        bin_arr[l*sec_size : l*sec_size+logM] = int_2_bin_arr(idxs2[l], logM)
        if K != 1:
            bin_arr[l*sec_size+logM : (l+1)*sec_size] = psk_demod(vals[l], K)

    return bin_arr

def msg_vector_mmse_estimator(s, tau, M, K=1):
    '''
    MMSE (Bayes optimal) estimator of message vector of SPARC in
    (possibly complex) independent additive Gaussian noise.

    s  : effective observation in Gaussian noise (length L*M vector)
    tau: the noise variance (length L*M vector)
    L  : number of sections (1 non-zero entry per section)
    M  : number of entries per section
    K  : number of possible non-zero values (for K-PSK modulated SPARCs)
    '''
    assert type(s)==np.ndarray
    assert s.size % M==0
    assert type(tau)==float or tau.dtype==np.float

    if np.iscomplexobj(s):
        tau /= 2 # In order to reuse the eta function in the real case

    # The current method of preventing overflow is:
    # 1) subtract the maximum entry of array x from all entries before taking
    #    np.exp(x). Note: the largest exponent that np.float64 can handle is
    #    roughly 709.
    # 2) Use np.float128.
    #
    # Perhaps I can avoid tau becoming too small by changing how I do early
    # termination in the AMP algorithm.

    if K==1:
        x   = s.real / tau
        top = np.exp(x - x.max(), dtype=np.float128)
        bot = top.reshape(-1, M).sum(axis=1).repeat(M)
    elif K==2:
        x   = s.real / tau
        #top = np.sinh(x) # Can't use this due to overflow
        #bot = np.cosh(x).reshape(-1, M).sum(axis=1).repeat(M)
        x_max = np.abs(x).max()
        tmp1  = np.exp( x-x_max,dtype=np.float128)
        tmp2  = np.exp(-x-x_max,dtype=np.float128)
        top   = tmp1 - tmp2
        bot   = (tmp1 + tmp2).reshape(-1, M).sum(axis=1).repeat(M)
    elif K==4:
        x   = s.real / tau
        y   = s.imag / tau
        #top = np.sinh(x) + 1j*np.sinh(y) # Can't use this due to overflow
        #bot = (np.cosh(x) + np.cosh(y)).reshape(-1, M).sum(axis=1).repeat(M)
        xy_max = np.maximum(np.abs(x),np.abs(y)).max()
        tmpx1  = np.exp( x-xy_max,dtype=np.float128)
        tmpx2  = np.exp(-x-xy_max,dtype=np.float128)
        tmpy1  = np.exp( y-xy_max,dtype=np.float128)
        tmpy2  = np.exp(-y-xy_max,dtype=np.float128)
        top    = (tmpx1-tmpx2) + 1j*(tmpy1-tmpy2)
        bot    = ((tmpx1+tmpx2)+(tmpy1+tmpy2)).reshape(-1, M).sum(axis=1).repeat(M)
    else:
        c   = psk_constel(K)
        x   = np.outer(s/tau, c.conj()).real
        tmp = np.exp(x-np.abs(x).max(), dtype=np.float128)
        top = np.dot(tmp, c)
        bot = tmp.sum(axis=1).reshape(-1, M).sum(axis=1).repeat(M)

    # Cast back to normal float or complex data types
    if K==1 or K==2:
        return (top / bot).astype(np.float)
    else:
        return (top / bot).astype(np.complex)

def msg_vector_map_estimator(s, M, K=1):
    '''
    MAP estimator of message vector of SPARC in independent
    (possibly complex) additive white Gaussian noise.

    s  : effective observation in Gaussian noise (length L*M vector)
    M  : number of entries per section
    K  : number of possible non-zero values (for K-PSK modulated SPARCs)
    '''
    assert type(s)==np.ndarray
    assert s.size % M==0
    L = s.size // M # Number of sections

    if K==1 or K==2:
        beta = np.zeros_like(s, dtype=float).reshape(L,-1)
    else:
        beta = np.zeros_like(s, dtype=complex).reshape(L,-1)

    if K==1:
        idxs = s.real.reshape(L,-1).argmax(axis=1)
        beta[np.arange(L), idxs] = 1
    elif K==2:
        s2   = s.real.reshape(L,-1)
        idxs = np.abs(s2).argmax(axis=1)
        sgns = np.sign(s2[np.arange(L), idxs])
        beta[np.arange(L), idxs] = sgns
    elif K==4:
        s2   = np.maximum(np.abs(s.real), np.abs(s.imag)).reshape(L,-1)
        idxs = s2.argmax(axis=1)
        agls = np.angle(s.reshape(L,-1)[np.arange(L), idxs]) # Angles
        k    = np.rint(K*agls/(2*np.pi)).astype(int) # Round to integer idx
        k[np.where(k<0)] += K                   # Shift -ve ones by period K
        c    = psk_constel(K)
        beta[np.arange(L), idxs] = c[k]
    else:
        # Could've used this for K=2 and K=4, but this is much slower.
        # Approx. 5x-10x slower for K=2 and 2x slower for K=4.
        # For each section, finds the idx of the maximum of an M-by-K matrix.
        c = psk_constel(K)
        for l in range(L):
            s_l = s[l*M:(l+1)*M]
            tmp = np.outer(s_l.conj(), c).real
            idx1, idx2 = np.unravel_index(tmp.argmax(), tmp.shape)
            beta[l, idx1] = c[idx2]

    return beta.ravel()

######## Base matrix design operations ########

def pa_iterative(P, sigmaSqr, B, R_PA):
    ''' Iterative power allocation based on asymptotic SE.

    Has a factor of L difference compared with Adam's.
    P is the mean of all powers.
    '''
    Q = np.zeros(B)
    for b in range(B):
        phi     = sigmaSqr + P - Q.mean()
        Pblock  = 2 * np.log(2) * R_PA * phi
        Pspread = (B*P - Q.sum()) / (B - b)
        if Pblock > Pspread:
            Q[b : b+1] = Pblock
        else:
            Q[b:] = Pspread
            break
    Q /= Q.mean()/P # In case Q isn't normalised (e.g. with small B)
    return Q

def sc_basic(Q, omega, Lambda):
    '''
    Construct (omega, Lambda) spatially coupled base matrix
    with uncoupled base entry/vector Q.

    Q     : an np.ndarray
            1) base entry np.array(P) if regular SPARC, or
            2) base vector (power allocation) of length B
    omega : coupling width
    Lambda: coupling length
    '''

    assert type(Q) == np.ndarray

    if Q.ndim == 0: # No power allocation
        Lr = Lambda + omega - 1
        Lc = Lambda
        W_rc = Q*Lr/omega
        W    = np.zeros((Lr, Lc))
        for c in range(Lc):
            W[c : c+omega, c] = W_rc
    elif Q.ndim == 1: # With power allocation
        B  = Q.size
        Lr = Lambda + omega - 1
        Lc = Lambda * B
        W  = np.zeros((Lr, Lc))
        for c in range(Lambda):
            for r in range(c, c+omega):
                W[r, c*B :(c+1)*B] = Q*Lr/omega
    else:
        raise Exception('Something wrong with Q')

    assert np.isclose(W.mean(),np.mean(Q)),"Average base matrix values must equal P"
    return W

def create_base_matrix(P, power_allocated=False, spatially_coupled=False, **kwargs):
    '''
    Construct base entry/vector/matrix for Sparse Regression Codes

    For  power_allocated,  will need awgn_var, B, R and R_PA_ratio
    For spatially_coupled, will need omega and Lambda
    '''
    if not power_allocated:
        Q = np.array(P) # Make into np.ndarray to use .ndim==0
    else:
        awgn_var,B,R,R_PA_ratio = map(kwargs.get,['awgn_var','B','R','R_PA_ratio'])
        Q = pa_iterative(P, awgn_var, B, R*R_PA_ratio)

    if not spatially_coupled:
        W = Q
    else:
        omega, Lambda = map(kwargs.get,['omega','Lambda'])
        W = sc_basic(Q, omega, Lambda)

    return W

######## DCT/FFT operations ########

def sub_fft(m, n, seed=0, order0=None, order1=None):
    """
    Returns functions to compute the sub-sampled fast Fourier transform,
    i.e., matrix-vector multiply with subsampled rows from the DFT matrix.

    This is a direct modification of Adam Greig's pyfht source code which can
    be found at https://github.com/adamgreig/pyfht/blob/master/pyfht.py

    [Parameters]
    m: number of rows
    n: number of columns
    m < n
    Most efficient (but not required) for max(m+1,n+1) to be a power of 2.
    seed:   determines choice of random matrix
    order0: optional m-long array of row indices in [1, max(m+1,n+1)] to
            implement subsampling of rows; generated by seed if not specified.
    order1: optional n-long array of row indices in [1, max(m+1,n+1)] to
            implement subsampling of columns; generated by seed if not specified.

    [Returns]
    Ax(x):    computes A.x (of length m), with x having length n
    Ay(y):    computes A*.y (of length n), with y having length m
    """
    assert type(m)==int and m>0
    assert type(n)==int and n>0
    w = 2**int(np.ceil(np.log2(max(m+2,n+2))))

    if order0 is not None and order1 is not None:
        assert order0.shape == (m,)
        assert order1.shape == (n,)
    else:
        rng = np.random.RandomState(seed)
        idxs0 = np.delete(np.arange(w, dtype=np.uint32), [0, w//2])
        idxs1 = np.delete(np.arange(w, dtype=np.uint32), [0, w//2])
        rng.shuffle(idxs0)
        rng.shuffle(idxs1)
        order0 = idxs0[:m]
        order1 = idxs1[:n]

    def Ax(x):
        assert x.size == n, "x must be n long"
        x_ext = np.zeros(w, dtype=complex)
        x_ext[order1] = x.reshape(n)
        y = np.fft.fft(x_ext)
        return y[order0]

    def Ay(y):
        assert y.size == m, "input must be m long"
        y_ext = np.zeros(w, dtype=complex)
        y_ext[order0] = y
        x_ext = np.fft.fft(y_ext.conj()).conj()
        return x_ext[order1]

    return Ax, Ay

def sub_dct(m, n, seed=0, order0=None, order1=None):
    """
    Returns functions to compute the sub-sampled fast Fourier transform,
    i.e., matrix-vector multiply with subsampled rows from the DFT matrix.

    This is a direct modification of Adam Greig's pyfht source code which can
    be found at https://github.com/adamgreig/pyfht/blob/master/pyfht.py

    [Parameters]
    m: number of rows
    n: number of columns
    m < n
    Most efficient (but not required) for max(m+1,n+1) to be a power of 2.
    seed:   determines choice of random matrix
    order0: optional m-long array of row indices in [1, max(m+1,n+1)] to
            implement subsampling of rows; generated by seed if not specified.
    order1: optional n-long array of row indices in [1, max(m+1,n+1)] to
            implement subsampling of columns; generated by seed if not specified.

    [Returns]
    Ax(x):    computes A.x (of length m), with x having length n
    Ay(y):    computes A*.y (of length n), with y having length m
    """
    assert type(m)==int and m>0
    assert type(n)==int and n>0
    w = 2**int(np.ceil(np.log2(max(m+1,n+1))))

    if order0 is not None and order1 is not None:
        assert order0.shape == (m,)
        assert order1.shape == (n,)
    else:
        rng = np.random.RandomState(seed)
        idxs0 = np.arange(1, w, dtype=np.uint32)
        idxs1 = np.arange(1, w, dtype=np.uint32)
        rng.shuffle(idxs0)
        rng.shuffle(idxs1)
        order0 = idxs0[:m]
        order1 = idxs1[:n]

    def Ax(x):
        assert x.size == n, "x must be n long"
        x_ext = np.zeros(w)
        x_ext[order1] = x.reshape(n)
        y = np.sqrt(w)*dct(x_ext, norm='ortho')
        return y[order0]

    def Ay(y):
        assert y.size == m, "input must be m long"
        y_ext = np.zeros(w)
        y_ext[order0] = y
        x_ext = np.sqrt(w)*idct(y_ext, norm='ortho')
        return x_ext[order1]

    return Ax, Ay

def sparc_transforms(W, L, M, n, rand_seed, csparc=False):
    """
    Construct two functions to compute matrix-vector multiplications with a
    Hadamard design matrix `A` using the Fast Walsh-Hadamard Transform (FWHT),
    or with a Fourier design matrix using the fast Fourier Transform (FFT).

    W: an np.ndarray which determines design matrix variance structure
       1) regular base entry (=P)
       2) base vector of size B
       3) base matrix of size Lr-by-Lc
    Mr: number of  rows   per block of the design matrix
    Mc: number of columns per block of the design matrix
    L: number of sections in the message vector
    rand_seed: determines choice of random matrix
    csparc: whether or not we are using complex SPARCs.

    Splits the overall matrix-vector multiplication into smaller blocks (may
    only have one block if W.ndim==0 of size `Mr` by `Mc` each. Uses `sub_fht`
    to compute the FWHT of each block. Scale each block by a factor of
    sqrt(W[r,c]/L).

    Most efficient (but not required) when max(Mr+1,Mc+1) is a power of two.

    Returns
        Ab(x): computes `A` times `x` (`x` has length L*M)
        Az(y): computes `A` transpose times `y` (`y` has length n)
    """

    assert type(W)==np.ndarray
    assert type(L)==int and type(M)==int and type(n)==int
    assert L>0 and M>0 and n>0

    def generate_ordering(W, Mr, Mc, rand_seed, csparc):
        '''
        Generate random ordering for SPARC transforms
        '''

        W_shape = W.shape
        order0  = np.zeros(W_shape + (Mr,), dtype=np.uint32) # Row order
        order1  = np.zeros(W_shape + (Mc,), dtype=np.uint32) # Column order
        if not csparc:
            w = 2**int(np.ceil(np.log2(max(Mr+1, Mc+1)))) # Transform size
            idxs0 = np.arange(1, w, dtype=np.uint32)      # Avoid 1st row
            idxs1 = np.arange(1, w, dtype=np.uint32)      # Avoid 1st column
        else:
            w = 2**int(np.ceil(np.log2(max(Mr+2, Mc+2))))
            idxs0 = np.delete(np.arange(w, dtype=np.uint32), [0,w//2])
            idxs1 = np.delete(np.arange(w, dtype=np.uint32), [0,w//2])

        rng = np.random.RandomState(rand_seed)
        if W.ndim == 0:
            rng.shuffle(idxs0)
            rng.shuffle(idxs1)
            order0 = idxs0[:Mr]
            order1 = idxs1[:Mc]
        elif W.ndim == 1:
            for b in range(W_shape[0]):
                rng.shuffle(idxs0)
                rng.shuffle(idxs1)
                order0[b] = idxs0[:Mr]
                order1[b] = idxs1[:Mc]
        elif W.ndim == 2:
            for r in range(W_shape[0]):
                for c in range(W_shape[1]):
                    if W[r,c] != 0:
                        rng.shuffle(idxs0)
                        rng.shuffle(idxs1)
                        order0[r,c] = idxs0[:Mr]
                        order1[r,c] = idxs1[:Mc]
        else:
            raise Exception("Something is wrong with the ordering")

        return order0, order1

    if W.ndim == 0:
        Mr = n
        Mc = L*M
        order0, order1 = generate_ordering(W, Mr, Mc, rand_seed, csparc)
        if not csparc:
            ax, ay = sub_dct(Mr, Mc, order0=order0, order1=order1)
        else:
            ax, ay = sub_fft(Mr, Mc, order0=order0, order1=order1)

        def Ab(x):
            assert x.size == Mc
            out = np.sqrt(W/L)*ax(x)
            return out

        def Az(y):
            assert y.size == Mr
            out = np.sqrt(W/L)*ay(y)
            return out

    elif W.ndim == 1:
        B  = W.size # Number of blocks
        assert L % B == 0
        Mr = n
        Mc = L*M // B
        order0, order1 = generate_ordering(W, Mr, Mc, rand_seed, csparc)

        ax = np.empty(B, dtype=np.object)
        ay = np.empty(B, dtype=np.object)
        for b in range(B):
            if not csparc:
                ax[b], ay[b] = sub_dct(Mr, Mc, order0=order0[b], order1=order1[b])
            else:
                ax[b], ay[b] = sub_fft(Mr, Mc, order0=order0[b], order1=order1[b])

        def Ab(x):
            assert x.size == B*Mc
            if not csparc:
                out = np.zeros(Mr)
            else:
                out = np.zeros(Mr, dtype=complex)
            for b in range(B):
                out += np.sqrt(W[b]/L)*ax[b](x[b*Mc:(b+1)*Mc])
            return out

        def Az(y):
            assert y.size == Mr
            if not csparc:
                out = np.zeros(B*Mc)
            else:
                out = np.zeros(B*Mc, dtype=complex)
            for b in range(B):
                out[b*Mc:(b+1)*Mc] += np.sqrt(W[b]/L)*ay[b](y)
            return out

    elif W.ndim == 2:
        Lr, Lc = W.shape
        assert L % Lc == 0
        assert n % Lr == 0
        Mc     = L*M // Lc
        Mr     = n // Lr
        order0, order1 = generate_ordering(W, Mr, Mc, rand_seed, csparc)

        ax = np.empty((Lr,Lc), dtype=np.object)
        ay = np.empty((Lr,Lc), dtype=np.object)
        for r in range(Lr):
            for c in range(Lc):
                if W[r,c] != 0:
                    if not csparc:
                        ax[r,c], ay[r,c] = sub_dct(Mr, Mc, order0=order0[r,c],
                                                           order1=order1[r,c])
                    else:
                        ax[r,c], ay[r,c] = sub_fft(Mr, Mc, order0=order0[r,c],
                                                           order1=order1[r,c])

        def Ab(x):
            assert x.size == Lc*Mc
            if not csparc:
                out = np.zeros(Lr*Mr)
            else:
                out = np.zeros(Lr*Mr, dtype=complex)
            for r in range(Lr):
                for c in range(Lc):
                    if W[r,c] != 0:
                        out[r*Mr:(r+1)*Mr] += (np.sqrt(W[r,c]/L)*
                                               ax[r,c](x[c*Mc:(c+1)*Mc]))
            return out

        def Az(y):
            assert y.size == Lr*Mr
            if not csparc:
                out = np.zeros(Lc*Mc)
            else:
                out = np.zeros(Lc*Mc, dtype=complex)
            for r in range(Lr):
                for c in range(Lc):
                   if W[r,c] != 0:
                        out[c*Mc:(c+1)*Mc] += (np.sqrt(W[r,c]/L)*
                                               ay[r,c](y[r*Mr:(r+1)*Mr]))
            return out

    else:
        raise Exception('Something wrong with base matrix input W')

    return Ab, Az

######## AMP decoder ########
def sparc_amp(y, code_params, decode_params, awgn_var, rand_seed, beta0, Ab=None, Az=None):
    """
    AMP decoder for Spatially Coupled Sparse Regression Codes

    y: received (noisy) output symbols
    awgn_var: awgn channel noise variance
    beta0: true message vector, only used to calculate NMSE.
    Ab(x): computes design matrix `A` times `x` (`x` has length L*M)
    Az(y): computes `A` transpose times `y` (`y` has length n)
    """

    # Get code parameters
    P,R,L,M,n = map(code_params.get, ['P','R','L','M','n'])
    K = code_params['K'] if code_params['modulated'] else 1

    # Construct base matrix
    tmp = code_params.copy()
    tmp.update({'awgn_var':awgn_var})
    W = create_base_matrix(**tmp)
    assert 0 <= W.ndim <= 2

    # Get decode parameters
    t_max, rtol, phi_est_method = map(decode_params.get,
                                      ['t_max','rtol','phi_est_method'])
    assert phi_est_method==1 or phi_est_method==2

    # Functions to calculate (A * beta) and (A.T * z) if needed
    if (Ab is None) or (Az is None):
        Ab, Az = sparc_transforms(W, L, M, n, rand_seed, code_params['complex'])

    # Initialise variables
    beta = np.zeros(L*M) if (K==1 or K==2) else np.zeros(L*M, dtype=complex)
    z    = y                               # Residual (modified) vector
    atol = 2*np.finfo(np.float).resolution # abs tolerance 4 early stopping
    if W.ndim == 0:
        gamma = W
        nmse  = np.ones(t_max)
    else:
        if W.ndim == 2:
            Lr = W.shape[0]               # Num of row blocks
            Mr = n // Lr                  # Entries per row block
        Lc    = W.shape[-1]               # Num of column blocks
        Mc    = L*M // Lc                 # Entries per column block
        gamma = np.dot(W, np.ones(Lc))/Lc # Residual var - noise var (length Lr)
        nmse  = np.ones((t_max, Lc))      # NMSE of each column block


    # Run AMP decoder
    for t in range(t_max-1):
        if t > 0:
            psi_prev = np.copy(psi)
            phi_prev = np.copy(phi)

            if W.ndim == 0:
                gamma = W * psi # approx residual_var - noise_var
            else:
                gamma = np.dot(W, psi)/Lc

            # Modified residual z
            b = gamma/phi_prev # Length Lr
            if W.ndim != 2:
                z = y - Ab(beta) + b*z
            else:
                z = y - Ab(beta) + b.repeat(Mr)*z

        # Residual variance phi
        if phi_est_method == 1:
            phi = awgn_var + gamma
        elif phi_est_method == 2:
            if W.ndim != 2:
                phi = (np.abs(z)**2).mean()
            else:
                phi = (np.abs(z)**2).reshape(Lr,-1).mean(axis=1)

        # Effective noise variance tau
        if W.ndim == 0:
            tau     = (L*phi/n)/W # Scalar
            tau_use = tau         # Scalar
            phi_use = phi         # Scalar
        elif W.ndim == 1:
            tau     = (L*phi/n)/W    # Length Lc
            tau_use = tau.repeat(Mc) # Length LM
            phi_use = phi            # Scalar
        elif W.ndim == 2:
            tau     = (L/Mr)/np.dot(W.T,1/phi) # Length Lc
            tau_use = tau.repeat(Mc)           # Length LM
            phi_use = phi.repeat(Mr)           # Length n

        # Update message vector beta
        s    = beta + tau_use * Az(z/phi_use)
        beta = msg_vector_mmse_estimator(s, tau_use, M, K)

        # Update NMSE and estimate of NMSE
        if W.ndim == 0:
            psi       = 1 - (np.abs(beta)**2).sum()/L
            nmse[t+1] = (np.abs(beta-beta0)**2).sum()/L
        else:
            psi       = 1 - (np.abs(beta)**2).reshape(Lc,-1).sum(axis=1)/(L/Lc)
            nmse[t+1] = (np.abs(beta-beta0)**2).reshape(Lc,-1).sum(axis=1)/(L/Lc)

        # Early stopping criteria
        if t>0 and np.allclose(psi, psi_prev, rtol, atol=atol):
            nmse[t:] = nmse[t]
            break

    t_final = t+1

    # Obtain final beta estimate by doing hard decision
    # Hard decision is done on s and not beta NOT on because s has the correct
    # distributional property (true beta + Gaussian noise).
    # So then doing hard decision on s is actually doing MAP estimation.
    # Recall that beta^{t+1} is the MMSE estimator. Therefore, we can see this as
    # doing MAP instead of MMSE in the last iteration.
    # The BER analysis using SE parameter tau also assumes s is used.
    beta = msg_vector_map_estimator(s, M, K)

    return beta, t_final, nmse, psi

########################### TESTS ###########################

def test_bin_arr_msg_vector(k=1024*9, M=2**9):
    seed = list(np.random.randint(2**32-1, size=2))
    bin_array = rnd_bin_arr(k, seed)
    msg_vector = bin_arr_2_msg_vector(bin_array, M)
    bin_array_2 = msg_vector_2_bin_arr(msg_vector, M)
    assert np.array_equal(bin_array, bin_array_2)
