function fk=dirft1d1(nj,xj,cj,iflag,ms)
%DIRFT1D1: Direct (slow) computation of nonuniform FFT in R^1 - Type 1.
%
%  FK = DIRFT1D1(NJ,XJ,CJ,IFLAG,MS);
%
%               1  nj
%     fk(k1) = -- SUM cj(j) exp(+/-i k1 xj(j)) 
%              nj j=1
%
%     for -ms/2 <= k1 <= (ms-1)/2
%
%     If (iflag .ge.0) the + sign is used in the exponential.
%     If (iflag .lt.0) the - sign is used in the exponential.
%
%  Input parameters:
%
%     nj     number of sources   (integer)
%     xj     location of sources (real *8)
%
%            on interval [-pi,pi].
%
%     cj     strengths of sources (complex *16)
%     iflag  determines sign of FFT (see above)
%     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
%                 
%  Output parameters:
%
%     fk     Fourier transform values (complex *16)
%
%

fk=zeros(ms,1)+1i*zeros(ms,1);

mex_id_ = 'dirft1d1(i int[x], i double[], i dcomplex[], i int[x], i int[x], io dcomplex[])';
[fk] = nufft1d(mex_id_, nj, xj, cj, iflag, ms, fk, 1, 1, 1);


