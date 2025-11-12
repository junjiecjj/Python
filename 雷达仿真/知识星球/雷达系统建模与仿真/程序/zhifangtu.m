function zhifangtu(x,m)
%画数据的直方图,x表示要画的随机数,m表示所要画的条数
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a=min(x);
b=max(x);
l=length(x);
h=(b-a)/m;
%量化x
x=x/h;
x=ceil(x);
w=zeros(1,m);
for i=1:l
    for j=1:m
        if (x(i)==j)
%x(i)落在j的区间上，则w(j)加1            
            w(j)=w(j)+1;
        else 
           continue
        end
    end
end
w=w/(h*l);
z=a:h:(b-h);
bar(z,w);
title('直方图')