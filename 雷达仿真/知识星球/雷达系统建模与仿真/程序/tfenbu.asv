function y=tfenbu(m,n)
%本函数产生自由度为m，数据量为n的t分布。
%-----------------------------------
%
y=zeros(1,n);
if(floor(m/2)==m/2)
    for i=1:m/2
       [x1,x2]=gaussian(n);
       for j=1:n
        y(j)=x1(j)^2+x2(j)^2+y(j);
       end
    end
    z=gaussian(n);
    y=z./y;
else
    for i=1:floor(m/2)
       [x1,x2]=gaussian(n);
       for j=1:n
        y(j)=x1(j)^2+x2(j)^2+y(j);
       end
    end
    [x,z]=gaussian(n);
    for j=1:n
        y(j)=y(j)+x(j)^2;
    end
    y=z./y;
end