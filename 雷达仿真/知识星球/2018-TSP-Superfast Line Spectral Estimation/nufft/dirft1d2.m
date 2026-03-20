function cj=dirft1d2(nj,xj,iflag,ms,fk)
%DIRFT1D2: Direct (slow) computation of nonuniform FFT in R^1 - Type 2.
%
%  CJ = DIRFT1D2(NJ,XJ,IFLAG,MS,FK);
%
%     cj(j) = SUM   fk(k1) exp(+/-i k1 xj(j)) 
%             k1  
%                            for j = 1,...,nj
%
%     where -ms/2 <= k1 <= (ms-1)/2
%
%     If (iflag .ge.0) the + sign is used in the exponential.
%     If (iflag .lt.0) the - sign is used in the exponential.
%
%  Input parameters:
%
%     nj     number of output values   (integer)
%     xj     location of output values (real *8 array)
%     iflag  determines sign of FFT (see above)
%     ms     number of Fourier modes given  [ -ms/2: (ms-1)/2 ]
%     fk     Fourier coefficient values (complex *16 array)
%
%  Output parameters:
%
%     cj     output values (complex *16 array)
%
%

cj=zeros(nj,1)+1i*zeros(nj,1);

mex_id_ = 'dirft1d2(i int[x], i double[], io dcomplex[], i int[x], i int[x], i dcomplex[])';
[cj] = nufft1d(mex_id_, nj, xj, cj, iflag, ms, fk, 1, 1, 1);


