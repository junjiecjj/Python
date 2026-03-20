nj = 10000;
ms = 21;
mt = 21;
mu = 21;

xj = sort((rand(nj,1)*2-1)*pi);
yj = sort((rand(nj,1)*2-1)*pi);
zj = sort((rand(nj,1)*2-1)*pi);
cj = randn(nj,1)+1i*randn(nj,1);

eps=1e-12;


iflag = +1;

tic
fk = dirft3d1(nj,xj,yj,zj,cj,iflag,ms,mt,mu);
toc

tic
fk1 = nufft3d1(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu);
toc

abs_error=norm(fk-fk1,2)
rel_error=norm(fk-fk1,2)/norm(fk,2)


iflag = -1;

tic
fk = dirft3d1(nj,xj,yj,zj,cj,iflag,ms,mt,mu);
toc

tic
fk1 = nufft3d1(nj,xj,yj,zj,cj,iflag,eps,ms,mt,mu);
toc

abs_error=norm(fk-fk1,2)
rel_error=norm(fk-fk1,2)/norm(fk,2)
