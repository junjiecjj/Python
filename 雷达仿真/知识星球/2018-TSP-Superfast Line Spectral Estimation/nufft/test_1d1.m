nj = 10000;
ms = 21000;

xj = sort((rand(nj,1)*2-1)*pi);
cj = randn(nj,1)+1i*randn(nj,1);

eps=1e-12;


iflag = +1;

tic
fk = dirft1d1(nj,xj,cj,iflag,ms);
toc

tic
fk1 = nufft1d1(nj,xj,cj,iflag,eps,ms);
toc

abs_error=norm(fk-fk1,2)
rel_error=norm(fk-fk1,2)/norm(fk,2)


iflag = -1;

tic
fk = dirft1d1(nj,xj,cj,iflag,ms);
toc

tic
fk1 = nufft1d1(nj,xj,cj,iflag,eps,ms);
toc

abs_error=norm(fk-fk1,2)
rel_error=norm(fk-fk1,2)/norm(fk,2)
