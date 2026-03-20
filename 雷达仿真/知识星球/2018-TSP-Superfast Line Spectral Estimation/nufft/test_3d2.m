nj = 10000;
ms = 21;
mt = 21;
mu = 21;

xj = sort((rand(nj,1)*2-1)*pi);
yj = sort((rand(nj,1)*2-1)*pi);
zj = sort((rand(nj,1)*2-1)*pi);
fk = randn(ms,mt,mu)+1i*randn(ms,mt,mu);

eps = 1e-12;


iflag = +1;

tic
cj = dirft3d2(nj,xj,yj,zj,iflag,ms,mt,mu,fk);
toc

tic
cj1 = nufft3d2(nj,xj,yj,zj,iflag,eps,ms,mt,mu,fk);
toc

abs_error=norm(cj-cj1,2)
rel_error=norm(cj-cj1,2)/norm(cj,2)


iflag = -1;

tic
cj = dirft3d2(nj,xj,yj,zj,iflag,ms,mt,mu,fk);
toc

tic
cj1 = nufft3d2(nj,xj,yj,zj,iflag,eps,ms,mt,mu,fk);
toc

abs_error=norm(cj-cj1,2)
rel_error=norm(cj-cj1,2)/norm(cj,2)
