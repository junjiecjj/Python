function cj=dirft3d2(nj,xj,yj,zj,iflag,ms,mt,mu,fk)
%DIRFT3D2: Direct (slow) computation of nonuniform FFT in R^3 - Type 2.
%
%  CJ = DIRFT3D2(NJ,XJ,YJ,ZJ,IFLAG,MS,MT,MU,FK);
%
%     cj(j) = SUM      fk(k1,k2,k3) exp(+/-i (k1,k2,k3)*(xj(j),yj(j),zj(j)))
%             k1,k2,k3  
%                            for j = 1,...,nj
%
%     where -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, 
%           -mu/2 <= k3 <= (mu-1)/2
%
%     If (iflag .ge.0) the + sign is used in the exponential.
%     If (iflag .lt.0) the - sign is used in the exponential.
%
%  Input parameters:
%
%     nj     number of output values   (integer)
%     xj,yj,zj  location of output values (real *8 array)
%     iflag  determines sign of FFT (see above)
%     ms     number of Fourier modes given  [ -ms/2: (ms-1)/2 ]
%     mt     number of Fourier modes given  [ -mt/2: (mt-1)/2 ]
%     mu     number of Fourier modes given  [ -mt/2: (mu-1)/2 ]
%     fk     Fourier coefficient values (complex *16 array)
%
%  Output parameters:
%
%     cj     output values (complex *16 array)
%
%

cj=zeros(nj,1)+1i*zeros(nj,1);

mex_id_ = 'dirft3d2(i int[x], i double[], i double[], i double[], io dcomplex[], i int[x], i int[x], i int[x], i int[x], i dcomplex[])';
[cj] = nufft3d(mex_id_, nj, xj, yj, zj, cj, iflag, ms, mt, mu, fk, 1, 1, 1, 1, 1);


