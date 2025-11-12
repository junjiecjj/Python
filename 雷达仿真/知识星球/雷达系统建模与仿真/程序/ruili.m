function y=ruili(m,n)
%瑞利分布,m是瑞利分布的参数,n代表数据量,n一般要大于1024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=junyun(n);
for i=1:n
    if (x(i)==0)
        x(i)=0.0001;
    else 
        continue;
    end
end    
u=(-2)*log(x);
y=m*sqrt(u);