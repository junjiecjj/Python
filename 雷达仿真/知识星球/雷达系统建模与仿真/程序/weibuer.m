function y=weibuer(a,b,n)
%韦布尔分布,a,b表示参数,b不能为0.n表示数据量,一般要大于1024
%a=1时,是指数分布
%a=2时,是瑞利分布
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=junyun(n);
for i=1:n
    if (x(i)==0)
        x(i)=0.0001;
    else
        continue;
    end
end
u=-log(x);
y=b*u.^(1/a);