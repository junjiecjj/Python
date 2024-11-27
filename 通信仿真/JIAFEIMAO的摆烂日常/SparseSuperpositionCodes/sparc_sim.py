# Python code to run Sparse Regression Code (SPARCs) simulations
#
# Copyright (c) 2020 Kuan Hsieh

import numpy as np
from sparc import sparc_encode, sparc_decode

def sparc_sim(code_params, decode_params, awgn_var, rand_seed=None):

    """
    End-to-end simulation of Sparse Regression Code (SPARC) encoding/decoding
    in AWGN channel.
    """

    # Currently cheating as encoder directly passes fast transforms Ab and Az to
    # the deocder. (Decoder doesn't use random seed to generate fast transform.)

    # Simulation
    bits_i, beta0, x, Ab, Az = sparc_encode(code_params, awgn_var, rand_seed)
    y                        = awgn_channel(x, awgn_var, rand_seed)
    bits_o, beta, T, nmse, expect = sparc_decode(y, code_params, decode_params,
                                                 awgn_var, rand_seed, beta0, Ab, Az)

    # Error analysis
    ber     = calc_ber(bits_i, bits_o)
    cer     = 1.0*(ber>0)
    detect  = 1.0*(not (ber>0)^expect) # Frame error detected
    results = {'ber':ber, 'cer':cer, 't_final':T, 'nmse':nmse, 'detect':detect}

    # Section-wise error analysis
    if not code_params['modulated']:
        ser, loc_of_sec_errs, num_of_sec_errs = calc_ser(beta0, beta, code_params['L'])
        results_append = {'ser': ser,
                          'loc_of_sec_errs': loc_of_sec_errs,
                          'num_of_sec_errs': num_of_sec_errs}
    else:
        err_rates, loc_of_errs, num_of_errs = calc_ler_ver(beta0, beta, code_params['L'],
                                                                        code_params['K'])
        ler, ver, ser = err_rates
        loc_of_loc_errs, loc_of_val_errs, loc_of_sec_errs = loc_of_errs
        num_of_loc_errs, num_of_val_errs, num_of_sec_errs = num_of_errs
        results_append = {'ser': ser, 'ler': ler, 'ver': ver,
                          'loc_of_sec_errs': loc_of_sec_errs,
                          'loc_of_loc_errs': loc_of_loc_errs,
                          'loc_of_val_errs': loc_of_val_errs,
                          'num_of_sec_errs': num_of_sec_errs,
                          'num_of_loc_errs': num_of_loc_errs,
                          'num_of_val_errs': num_of_val_errs}
    results.update(results_append)

    # SNR parameters
    #snr = P/awgn_var              # Signal-to-noise ratio
    #C = 0.5 * np.log2(1 + snr)    # Channel capacity
    #EbN0 = 1/(2*R) * (P/awgn_var) # Energy per bit/noise PSD

    return results

######## Error analysis ########

def calc_ber(true_bin_array, est_bin_array):
    '''
    Calculate the bit error rate (BER)
    '''
    assert true_bin_array.dtype == 'bool'
    assert est_bin_array.dtype  == 'bool'
    k = true_bin_array.size
    assert k == est_bin_array.size
    return np.count_nonzero(np.bitwise_xor(true_bin_array, est_bin_array))/k

def calc_ser(beta0, beta, L):
    '''
    Find the section error rate of the estimated message vector.

    beta0 : true message vector
    beta  : estimated message vector
    L     : number of sections

    Returns
        ser: section error rate, i.e., # of sections decoded in error / L
        loc_of_sec_errs: list of integers specifying sections decoded in error
    '''
    assert beta.size == beta0.size,  'beta and beta0 are of different size'
    assert beta.dtype == beta0.dtype,'beta and beta0 are of different type'
    assert type(L)==int and L>0
    assert beta.size % L == 0

    M = beta.size // L
    error_array = np.zeros(L, dtype=bool)
    for l in range(L):
        error_array[l] = not np.array_equal(beta[l*M:(l+1)*M], beta0[l*M:(l+1)*M])

    num_of_sec_errs = np.count_nonzero(error_array)
    ser             = num_of_sec_errs / L
    loc_of_sec_errs = np.flatnonzero(error_array)

    return ser, loc_of_sec_errs, num_of_sec_errs

def calc_ler_ver(beta0, beta, L, K):
    '''
    Find the location error rate and value error rate of
    the estimated message vector.

    [Parameters]

    beta0 : true message vector
    beta  : estimated message vector
    L     : number of sections
    K     : number of K-PSK constellation symbols

    [Returns]

    ler: location error rate, i.e., # of nonzero locations in error / L
    ver: value error rate, i.e., # of nonzero values in error / L
    loc_of_loc_errs: list of integers specifying locations of location error
    loc_of_val_errs: list of integers specifying locations of value error

    ----------

    For modulated SPARCs, L*LogM bits are encoded in the location of the
    non-zero entries of the message vector whilst the other L*logK bits are
    encoded in the value of those non-zero entries.

    In addition to finding the fraction of message vector sections decoded
    in error, i.e. the section error rate (SER), we would like to find the
    fraction of non-zero entry locations were decoded in error, i.e.
    location error rate (LER), and also the fraction of non-zero entry values
    decoded in error, i.e. value error rate (VER).

    The LER is the same as the SER for non-modulated SPARCs – when a location
    is decoded in error, (on average) half of the bits are in error.
    The VER is different – if the non-zero value is decoded in error, it
    depends on whether the error is small or large. A small decoding error
    will result in a small bit error, e.g. decoding a symbol to its
    neighbouring constellation symbol would result in a single bit error with
    the use of Gray coding.

    '''
    assert beta.size == beta0.size,  'beta and beta0 are of different size'
    assert beta.dtype == beta0.dtype,'beta and beta0 are of different type'
    assert type(L)==int and L>0
    assert beta.size % L == 0

    M = beta.size // L

    ## Probably don't need for loop, reshape to L-by-M matrix
    beta0_reshape  = beta0.reshape(L,M)
    beta_reshape   = beta.reshape(L,M)
    idxs11, idxs12 = np.nonzero(beta0_reshape)
    idxs21, idxs22 = np.nonzero(beta_reshape)
    assert np.array_equal(idxs11, np.arange(L)) # Exactly 1 nonzero each row
    assert np.array_equal(idxs21, np.arange(L)) # Exactly 1 nonzero each row
    vals1 = beta0_reshape[(idxs11,idxs12)]
    vals2 = beta_reshape[(idxs21,idxs22)]

    loc_err = np.logical_not(idxs12 == idxs22)
    val_err = np.logical_not(vals1 == vals2)
    sec_err = np.logical_or(loc_err, val_err)

    num_of_loc_errs = np.count_nonzero(loc_err)
    num_of_val_errs = np.count_nonzero(val_err)
    num_of_sec_errs = np.count_nonzero(sec_err)
    ler = num_of_loc_errs / L
    ver = num_of_val_errs / L
    ser = num_of_sec_errs / L
    loc_of_loc_errs = np.flatnonzero(loc_err)
    loc_of_val_errs = np.flatnonzero(val_err)
    loc_of_sec_errs = np.flatnonzero(sec_err)

    num_of_errs = (num_of_loc_errs, num_of_val_errs, num_of_sec_errs)
    error_rates = (ler, ver, ser)
    loc_of_errs = (loc_of_loc_errs, loc_of_val_errs, loc_of_sec_errs)

    return error_rates, loc_of_errs, num_of_errs

######## Channel models ########

def awgn_channel(input_array, awgn_var, rand_seed):
    '''
    Adds Gaussian noise to input array

    Real input_array:
        Add Gaussian noise of mean 0 variance awgn_var.

    Complex input_array:
        Add complex Gaussian noise. Indenpendent Gaussian noise of mean 0
        variance awgn_var/2 to each dimension.
    '''

    assert input_array.ndim == 1, 'input array must be one-dimensional'
    assert awgn_var >= 0

    rng = np.random.RandomState(rand_seed)
    n   = input_array.size

    if input_array.dtype == np.float:
        return input_array + np.sqrt(awgn_var)*rng.randn(n)

    elif input_array.dtype == np.complex:
        return input_array + np.sqrt(awgn_var/2)*(rng.randn(n)+1j* rng.randn(n))

    else:
        raise Exception("Unknown input type '{}'".format(input_array.dtype))

