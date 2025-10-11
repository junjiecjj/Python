function  AF_m1 = AFmean_single_symbol( N_c,T_b,x,y,d)
%% 单个OFDM符号的平均模糊函数
% 理论值

% % 系统子载波数
% N_c=13;
% % 脉冲持续时间
% T_b=0.8e-6;
% % 归一化时间
% x=0:1:1;
% % 归一化频率
% y=-8:0.001:8;
[X,Y]=meshgrid(x,y);


% 子载波频率间隔
delta_f=1/T_b;
% 子载波频率
f=(0:N_c-1)./T_b;  
% 符号能量
E=T_b*sum(d);

% 单周期模糊函数均值
a1=T_b-abs(X*T_b);
b1=sinc(delta_f*Y.*a1);
c1=0;
d1=exp(1j*pi*delta_f*Y.*(T_b+X*T_b));
for k=1:N_c
    temp=exp(1j*2*pi*f(k)*X*T_b);
    c1=c1+d(k)*temp;
end
AF_m0=a1.*b1.*c1.*d1;
AF_m0=AF_m0/E;
AF_m1=abs(AF_m0);


end

