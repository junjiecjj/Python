function fk=dirft1d3(nj,xj,cj,iflag,nk,sk)
%DIRFT1D3: Direct (slow) computation of nonuniform FFT in R^1 - Type 3.
%
%  FK = DIRFT1D3(NJ,XJ,CJ,IFLAG,NK,SK);
%
%                 1  nj
%     fk(k)    = -- SUM cj(j) exp(+/-i s(k) xj(j)) 
%                nj j=1
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
%     nk     number of (noninteger) Fourier modes computed
%     sk     k-values (locations) of desired Fourier modes
%                 
%  Output parameters:
%
%     fk     Fourier transform values (complex *16)
%

fk=zeros(nk,1)+1i*zeros(nk,1);

mex_id_ = 'dirft1d3(i int[x], i double[], i dcomplex[], i int[x], i int[x], i double[], io dcomplex[])';
[fk] = nufft1d(mex_id_, nj, xj, cj, iflag, nk, sk, fk, 1, 1, 1);



