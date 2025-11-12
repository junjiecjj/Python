function y=bernoulli(p,n)
%产生数据量为n的贝努利分布,其中p属于(0-1)之间。
%-----------------------
%
u=junyun(n);
y=zeros(1,n);
for i=1:n
    if(u(i)<=p)
        y(i)=1;
    else
        y(i)=0;
    end
end