function y=swerling(n)
%swelingII·Ö²¼
%%%%%%%%%%%%%%%%%%%%%%
r=ones(1,n);
u=junyun(n);
v=junyun(n);
for i=1:n
    if (u(i)==0)
        u(i)=0.0001;
    else
        continue
    end
end
for i=1:n
    if (u(i)==v(i))
        u(i)=u(i)+0.0001
    else continue
    end
end    
t=-2*log(u);
h=2*pi*v;
x=sqrt(t).*cos(h);
z=sqrt(t).*sin(h);
y=(r/2).*(x.^2+z.^2);