function y=ffenbu(m1,m2,n)
%本函数产生自由度为(m1,m2)，数据量为n的F分布。
%-----------------------------------------
%
x1=kaifeng(m1,n);
x2=kaifeng(m2,n);
x1=x1/m1;
x2=x2/m2;
y=x1./x2;