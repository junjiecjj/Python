function y=duishuzhengtai(a,b,n)
%产生对数正态分布，a,b为随机分布的参数，n为数据量
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=gaussian(n);
u=sqrt(b)*x+a;
y=exp(u);