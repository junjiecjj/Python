nj = 10000;
ms = 21000;

xj = sort((rand(nj,1)*2-1)*pi);
fk = randn(ms,1)+1i*randn(ms,1);

eps = 1e-12;


iflag = +1;

tic
cj = dirft1d2(nj,xj,iflag,ms,fk);
toc

tic
cj1 = nufft1d2(nj,xj,iflag,eps,ms,fk);
toc

abs_error=norm(cj-cj1,2)
rel_error=norm(cj-cj1,2)/norm(cj,2)


iflag = -1;

tic
cj = dirft1d2(nj,xj,iflag,ms,fk);
toc

tic
cj1 = nufft1d2(nj,xj,iflag,eps,ms,fk);
toc

abs_error=norm(cj-cj1,2)
rel_error=norm(cj-cj1,2)/norm(cj,2)
