clc
clear all;
close all;

m=8 ;% array阵元
p=4; %  signal number信号数
N=3000;% recursive number 迭代次数 或快拍数
A=zeros(m,p); % array pattern阵列流型
theta=[30 0 -45 60]*pi/180;% the signal from the direction of 30 degree is expected. DOA 30为期望信号方向
j=sqrt(-1);
w=[0.01 0.2 0.3 0.4]*pi; % frequency for each signal.各个信号的数字频率

%%
s=to_get_s(w,N,p);
s_rec=get_s_rec(s,m,p,theta);
S=s_rec; %  output date matrix  .m*N 的阵元输出数据矩阵
%%%%——————————————%% 自适应调节权
y=S;% input data of GSC;
ad=exp(-j*pi*[0:m-1]'*sin(theta(1))) %steering vector in the direction of expected. 期望信号方向导向矢量
c=10;%  condition 波束形成条件
C=ad';
Wc=C'*inv(C*C')*c; %main path weight 主通道固定权
wa(1:m-1,1)=0;  % auxiliary path  辅助通道自适应权
B=get_B(m,theta); % get Block Matrix 得到阻塞矩阵
u=0.000001;
for k=1:N
    yb=conj(B)*y(:,k);  % m-1*1 的列向量
    Zc(k)=Wc.'*y(:,k);
    Za(k)=wa(:,k).'*yb;
    Z(k)=Zc(k)-Za(k);
    wa(:,k+1)=wa(:,k)-u*Z(k)*conj(yb);
end
%%%%------------
%%%main path 主通道
wop=Wc;
drawpp(m,wop);
%%%%auxiliary path 辅助通道
wop=B'*wa(:,N)
drawpp(m,wop);
%%array response 总的阵列响应
wop=Wc-B'*wa(:,N);
drawpp(m,wop);


function y=drawpp(m,wop)
    %figure,plot(e)
    thetas=[-90:90];
    tm=thetas*pi/180;
    am=exp(-j*pi*[0:m-1]'*sin(tm));
    A=abs(wop'*am);  %阵列响应
    A=A/max(A);
    figure,polar(tm,A)
    A=10*log10(A);  %对数图
    hold on,title('归一化阵列响应幅值极坐标图，八阵元，信噪比20db')
    figure,plot(thetas,A);
    hold on,title('八阵元，信噪比20db')
    hold on,xlabel('入射角/度')
    hold on,ylabel('归一化 A=10*log10(A);')
    grid on 
    axis([-90 90 -35 0]);
    hold on,plot(-45,-35:0.1:0,'r');
    hold on,plot(30,-35:0.1:0,'r');
    hold on,plot(0,-35:0.1:0,'r');
    hold on,plot(60,-35:0.1:0,'r');
end


function Bm=get_B(m,theta)  %用于产生阻塞矩阵%采用正交法构造阻塞矩阵
    u0=0.5*sin(theta(1)); % 假设阵元间距为半个波长
    a0=exp(-j*2*pi*[0:m-1]'*u0);
    u=u0+[1:m-1];
    B=exp(-j*2*pi*[0:m-1]'*u);
    Bm=conj(B');%% M-1*M 的矩阵
end

function Rxx=get_Rxx(s_rec,N,p,m)
    Rxx=zeros(m,m);
    x=1:m;
    for i=1:N
        for t=1:m
            x(t)=s_rec(t,i);
        end
        R=x'*x;
        Rxx=Rxx+R;
    end
    Rxx=Rxx/N;
end

function s_rec=get_s_rec(s,m,p,theta)  %用于产生经过阵元后的信号数据
    A=zeros(m,p);
    j=sqrt(-1);
    %%% 阵元间距为半个波长
    wi=pi*sin(theta);
    A=exp(-j*wi'*[0:(m-1)]);  % 阵列流型
    s_rec=A'*s;
    s_rec=awgn(s_rec,10);  % SNR=10 db
end

function s=to_get_s(w,N,p)
    s=zeros(p,N);
    for i=1:p
        s(i,1:N)=exp(j*w(i).*(1:N)); % 复指数信号  假设信道增益为 1
    end
end


