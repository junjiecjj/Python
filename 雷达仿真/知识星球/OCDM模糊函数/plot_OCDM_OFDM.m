%% OCDM与OFDM的模糊函数比较
clear all;close all;clc

% 子载波数
N_c=16;
% 脉冲持续时间
T_b=1e-6;
% 归一化时间
x=linspace(-1,1,32*N_c);%-1:0.001:1;
% 归一化频率
y=-5:0.01:5;
% 调频率
mu=16/T_b^2;
% 符号权重
% w=ones(N_c,1);
w=chebwin(N_c,50);
%% 模糊函数
[X,Y]=meshgrid(x,y);
[amf,amt]=size(X);
AF_ofdm = AFmean_single_symbol( N_c,T_b,x,y,w);
AF_ocdm = AFmean_lfm_OFDM_single( N_c,T_b,mu,x,y,w );

figure;grid on;hold on
surf(X,Y,AF_ofdm,'EdgeColor','none')
% title('OFDM平均模糊函数','fontsize',14)
xlabel('归一化时延','fontsize',14),ylabel('归一化频移','fontsize',14)
colormap('default')
% surface(x, [0,0], [zeros(1,amt);AF_ocdm(1,:)],'EdgeColor',[0 0 0],'FaceColor',[0 0 0]);
zlim([0,1]);
view(-46,26)


figure;grid on;hold on
surf(X,Y,AF_ocdm,'EdgeColor','none')
% title('OFDM平均模糊函数','fontsize',14)
xlabel('归一化时延','fontsize',14),ylabel('归一化频移','fontsize',14)
colormap('default')
% surface(x, [0,0], [zeros(1,amt);AF_ocdm(1,:)],'EdgeColor',[0 0 0],'FaceColor',[0 0 0]);
zlim([0,1]);
view(-46,26)
% view(0,90)
%% 距离模糊函数 
AC_ofdm=AF_ofdm(501,:);
AC_ocdm=AF_ocdm(501,:);

figure;grid on;hold on;
plot(x,AC_ofdm,'k','linewidth',1.5) 
plot(x,AC_ocdm,'b','linewidth',1.5) 
title('距离模糊函数比较','fontsize',14);
xlabel('归一化时延','fontsize',14),ylabel('归一化幅度','fontsize',14)
legend('OFDM','OCDM')
% ylim([0,1]);
% xlim([-0.2,0.2])
%% 速度模糊函数
DC_ofdm=AF_ofdm(:,256);
DC_ocdm=AF_ocdm(:,256);

figure;grid on;hold on;
plot(y,DC_ofdm,'k','linewidth',1.5) 
plot(y,DC_ocdm,'b','linewidth',1.5) 
title('速度模糊函数比较','fontsize',14);
xlabel('归一化频移','fontsize',14),ylabel('归一化幅度','fontsize',14)
legend('OFDM','OCDM')
% xlim([y(1),y(end)]);
% xlim([-5,5])



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

