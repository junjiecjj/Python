function y=zhishu(m,n)
%指数分布,m表示指数分布的参数,m不能为0.n表示数据量,n一般要大于1024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=junyun(n);
for i=1;n
    if (x(i)==0)
        x(i)=0.0001;
    else 
        continue;
    end
end    
u=log(x);
y=-(1/m)*u;