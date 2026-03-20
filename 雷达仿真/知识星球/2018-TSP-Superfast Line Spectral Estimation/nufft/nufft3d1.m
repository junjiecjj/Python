function [fk,ier]=nufft3d1(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu)
%NUFFT3D1: Nonuniform FFT in R^3 - Type 1.
%
%  [FK,IER] = NUFFT3D1(NJ,XJ,YJ,ZJ,CJ,IFLAG,EPS,MS,MT,MU);
%
%                     1  nj
%     fk(k1,k2,k3) = -- SUM cj(j) exp(+/-i (k1,k2,k3)*(xj(j),yj(j),zj(j)))   
%                    nj j=1 
%                                          for -ms/2 <= k1 <= (ms-1)/2
%                                          for -mt/2 <= k2 <= (mt-1)/2
%                                          for -mu/2 <= k3 <= (mu-1)/2
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
%     eps    precision request  (between 1.0e-15 and 1.0e-1)
%     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
%     mt     number of Fourier modes computed (-mt/2 to (mt-1)/2 )
%     mu     number of Fourier modes computed (-mu/2 to (mu-1)/2 )
%                 
%  Output parameters:
%
%     fk     Fourier transform values (complex *16)
%     ier    error return code   
%            ier = 0  => normal execution.
%            ier = 1  => precision eps requested is out of range.
%
%

fk=zeros(ms,mt,mu)+1i*zeros(ms,mt,mu);
ier=0;

mex_id_ = 'nufft3d1f90(i int[x], i double[], i double[], i double[], i dcomplex[], i int[x], i double[x], i int[x], i int[x], i int[x], io dcomplex[], io int[x])';
[fk, ier] = nufft3d(mex_id_, nj, xj, yj, zj, cj, iflag, eps, ms, mt, mu, fk, ier, 1, 1, 1, 1, 1, 1, 1);


