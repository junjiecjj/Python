function [ AF_u ] = AFmean_lfm_OFDM_single(  N_c,T_b,mu,x,y,weight )
%% lfm_ofdm单脉冲模糊函数均值
%% 调试参数
% 系统子载波数
% N_c=12;
% 脉冲持续时间
% T_b=0.8e-6;
% 调频斜率
% mu=20/T_b^2;
% 归一化时间
% x=-1:0.01:1;
% 归一化频率
% y=-10:0.01:10;
% 调制数据
% weight=ones(1,N_c);
% weight=chebwin(N_c,90);

%% 函数部分
[X,Y]=meshgrid(x,y);
% 子载波频率间隔
delta_f=1/T_b;
% 子载波频率
f=(0:N_c-1)./T_b;  
% 符号能量
E=T_b*sum(abs(weight));
[amf,amt]=size(X);
c1=zeros(amf,amt);
a1=T_b-abs(X*T_b);
b1=sinc((delta_f*Y+mu*X*T_b).*a1);
d1=exp(1j*pi*(delta_f*Y+mu*X*T_b).*(T_b+X*T_b));
for k=1:N_c
    temp=exp(1j*2*pi*f(k)*X*T_b-1j*pi*mu*(X*T_b).^2);
    c1=c1+weight(k)*temp;
end
AF_u=a1.*b1.*c1.*d1;
AF_u=AF_u/E;
AF_u=abs(AF_u);


% 模糊函数图
close all;
figure;grid on;hold on
surf(X,Y,AF_u,'EdgeColor','none')
title('lfm-OFDM模糊函数','fontsize',14)
xlabel('归一化时延','fontsize',14),ylabel('归一化频移','fontsize',14)
colormap('default')
% surface(x, [0,0], [zeros(1,amt);AF_u(1,:)],'EdgeColor',[0 0 0],'FaceColor',[0 0 0]);
% zlim([0,1]);view(0,90)

end




