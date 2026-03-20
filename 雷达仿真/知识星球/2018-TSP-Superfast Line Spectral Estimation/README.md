# Superfast Line Spectral Estimation
Implements the superfast method for line spectral estimation [1]. If you use
this code, please cite this work.

[1] T. L. Hansen, B. H. Fleury and B. D. Rao - "Superfast Line Spectral
Estimation", *IEEE Transactions on Signal Processing*, vol. 66, pp. 2511-2526, May 2018,
[preprint available on arXiv](https://arxiv.org/abs/1705.06073).

Abstract:
> A number of recent works have proposed to solve the line spectral
> estimation problem by applying off-the-grid extensions of sparse
> estimation techniques. These methods are preferable over classical line
> spectral estimation algorithms because they inherently estimate the model
> order. However, they all have computation times which grow at least
> cubically in the problem size, thus limiting their practical applicability
> in cases with large dimensions. To alleviate this issue, we
> propose a low-complexity method for line spectral estimation, which also
> draws on ideas from sparse estimation. Our method is based on a
> Bayesian view of the problem. The signal covariance matrix is shown to
> have Toeplitz structure, allowing superfast Toeplitz inversion to be used.
> We demonstrate that our method achieves estimation accuracy at least as
> good as current methods and that it does so while being orders of
> magnitudes faster.

## Setup & Usage
A significant speedup of the code is obtained by building a mex version of the
generalized Schur algorithm which is used internally. Do so by running
`buildmex` in the MATLAB prompt. The MATLAB codegen feature is used to generate
mex files.

To allow static memory allocation, the largest N = length(y) that the generated
mex files will support is specified in `buildmex.m`.

MATLAB did not support recursion in codegen prior to version 9.0 (R2016a). On
these earlier versions a, slower, fallback approach is used which only uses
mex for the innermost iteration of the generalized Schur algorithm.

A simple example is provided in `example.m`. Type `help superfast_lse` in a
MATLAB prompt for usage details.

This package redistributes the NUFFT codes from
http://www.cims.nyu.edu/cmcl/nufft/nufft.html.

