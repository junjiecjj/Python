function [v1,v2, v1tmp, v2tmp] = AsymRate(var, rho, b, vstar1, rho1, vstar2, rho2, v_star,v_s)

c_star = v_star^-1;
c=(1-b)*c_star;

%% 由于R1和R2是对称情况，故以下代码适合于b_ik很大时，当b_ik很小时需要对v2进行限制
vx=vstar2;
delta = (1+b-2*c.*vx).^2+8*b*c.*vx;
v1tmp=((2*c.*vx-1-b)+sqrt(delta))/(2*c);

for i = 1:length(vx)
    v1tmp(i) =min(v1tmp(i), NumMSEQspk(rho2(i)));
end
%v2tmp =(b.*(v1tmp.^-1)+c).^-1;
v2tmp = 2.*vx-v1tmp;
v1 =[vstar1, v1tmp];
v2 =[vstar1, v2tmp];



% % plot(rho,v_s,'green-');
% % hold on;
plot(rho,var,'r-');
hold on;
plot(rho,v_s,'-*')
plot(rho,v1,'black-');
hold on;
plot(rho,v2,'blue-');
%plot(rho2,v1tmp,'black-');
% hold on;
%plot(rho2,v2tmp,'blue-');
% % axis off
end

