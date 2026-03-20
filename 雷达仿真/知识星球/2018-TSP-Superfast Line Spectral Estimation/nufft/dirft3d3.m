function fk=dirft3d3(nj,xj,yj,zj,cj,iflag,nk,sk,tk,uk)
%DIRFT3D3: Direct (slow) computation of nonuniform FFT in R^3 - Type 3.
%
%  FK = DIRFT3D3(NJ,XJ,YJ,ZJ,CJ,IFLAG,NK,SK,TK,UK);
%
%                 1  nj
%     fk(k)    = -- SUM cj(j) exp(+/-i (s(k),t(k),u(k))*(xj(j),yj(j),zj(j)))
%                nj j=1
%
%     If (iflag .ge.0) the + sign is used in the exponential.
%     If (iflag .lt.0) the - sign is used in the exponential.
%
%  Input parameters:
%
%     nj     number of sources   (integer)
%     xj,yj,zj  location of sources (real *8)
%
%            on interval [-pi,pi].
%
%     cj     strengths of sources (complex *16)
%     iflag  determines sign of FFT (see above)
%     nk     number of (noninteger) Fourier modes computed
%     sk,tk,uk  k-values (locations) of desired Fourier modes
%                 
%  Output parameters:
%
%     fk     Fourier transform values (complex *16)
%

fk=zeros(nk,1)+1i*zeros(nk,1);

mex_id_ = 'dirft3d3(i int[x], i double[], i double[], i double[], i dcomplex[], i int[x], i int[x], i double[], i double[], i double[], io dcomplex[])';
[fk] = nufft3d(mex_id_, nj, xj, yj, zj, cj, iflag, nk, sk, tk, uk, fk, 1, 1, 1);


