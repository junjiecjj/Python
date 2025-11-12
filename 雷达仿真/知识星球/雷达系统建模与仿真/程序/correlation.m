function y=correlation(x)
%计算x的自相关函数
%---------------
%
n=length(x);
for i=1:n
    x1(i)=x(n+1-i);
end
y=conv(x,x1);