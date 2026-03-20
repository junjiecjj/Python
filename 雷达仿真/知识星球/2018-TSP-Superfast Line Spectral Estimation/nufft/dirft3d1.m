function fk=dirft3d1(nj,xj,yj,zj,cj,iflag,ms,mt,mu)
%DIRFT3D1: Direct (slow) computation of nonuniform FFT in R^3 - Type 1.
%
%  FK = DIRFT3D1(NJ,XJ,YJ,ZJ,CJ,IFLAG,MS,MT,MU);
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
%     ms     number of Fourier modes computed (-ms/2 to (ms-1)/2 )
%     mt     number of Fourier modes computed (-mt/2 to (mt-1)/2 )
%     mu     number of Fourier modes computed (-mu/2 to (mu-1)/2 )
%                 
%  Output parameters:
%
%     fk     Fourier transform values (complex *16)
%
%

fk=zeros(ms,mt,mu)+1i*zeros(ms,mt,mu);

mex_id_ = 'dirft3d1(i int[x], i double[], i double[], i double[], i dcomplex[], i int[x], i int[x], i int[x], i int[x], io dcomplex[])';
[fk] = nufft3d(mex_id_, nj, xj, yj, zj, cj, iflag, ms, mt, mu, fk, 1, 1, 1, 1, 1);


