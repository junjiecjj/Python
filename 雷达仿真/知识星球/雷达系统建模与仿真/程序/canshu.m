function y=canshu(x);
y=ones(1,2);
n=length(x);
y(1)=sum(x)/n;
z=x-y(1);
z=z.^2;
y(2)=sum(z)/(n-1);