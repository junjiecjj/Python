nj = 10000;
nk = 21000;

xj = sort((rand(nj,1)*2-1)*pi);
yj = sort((rand(nj,1)*2-1)*pi);
zj = sort((rand(nj,1)*2-1)*pi);
cj = randn(nj,1)+1i*randn(nj,1);
sk = sort((rand(nk,1)*2-1)*pi);
tk = sort((rand(nk,1)*2-1)*pi);
uk = sort((rand(nk,1)*2-1)*pi);

eps=1e-12;


iflag = +1;

tic
fk = dirft3d3(nj,xj,yj,zj,cj,iflag,nk,sk,tk,uk);
toc

tic
fk1 = nufft3d3(nj,xj,yj,zj,cj,iflag,eps,nk,sk,tk,uk);
toc

abs_error=norm(fk-fk1,2)
rel_error=norm(fk-fk1,2)/norm(fk,2)


iflag = -1;

tic
fk = dirft3d3(nj,xj,yj,zj,cj,iflag,nk,sk,tk,uk);
toc

tic
fk1 = nufft3d3(nj,xj,yj,zj,cj,iflag,eps,nk,sk,tk,uk);
toc

abs_error=norm(fk-fk1,2)
rel_error=norm(fk-fk1,2)/norm(fk,2)
