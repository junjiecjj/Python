function [t,s,g,f0,fs,f1]=huibo
%产生目标回波信号x,系统噪声y，地物杂波z以及回波p
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f0=3*10^7;      %发射信号频率
w=0;            %发射信号初始相位
c=3*10^8;       %光速
l=c/f0;         %雷达信号波长(载波波长)
R=40000;        %目标范围
Vd=200;         %雷达与目标之间的径向速度
fd=2*Vd/l;      %多普勒频率
Tr=600/f0;      %脉冲重复周期
N=10;           %雷达脉冲串长度
f1=f0/5;       %调频带宽是发射信号频率的1/5
k=10*f1/Tr;
fs=3*f0;        %仿真采样频率
Ts=1/fs;
%Tt=2*R/c;
Btar=4*pi*R/l;  %
M=floor(Tr*fs); %一个脉冲重复周期内的采样点数M=600
mt1=floor(2*Tr*fs/5);%mt1=720
mt2=floor(3*Tr*fs/10);%mt2=540
mt3=floor(7*Tr*fs/10);%mt3=1260
mt4=floor(3*Tr*fs/5);%mt4=1080
mt5=mt1-mt2;%mt5=180
Vgain=6;
s=zeros(M,N); %回波幅度起伏 
for m=1:M
    for n=1:N
        v(m,n)=(u(mt1-m)-u(mt2-m))*Vgain*cos(2*pi*f0*(m-mt2)*Ts+2*pi*k*Ts*(m-mt2)/2*(m-mt2)*Ts+2*pi*fd*n*Tr);
        g(m,n)=(u(mt3-m)-u(mt4-m))*Vgain*cos(2*pi*(f0*(m-mt4)*Ts+k*Ts*(m-mt4)/2*(m-mt4)*Ts));
        %u(t)是发射信号包络
    end
end
for i=1:M/10
    t(i)=Vgain*cos(2*pi*(k*Ts*(M/10-i)/2*(M/10-i)*Ts));
end
for n=1:N
    y(1:M,n)=gaussian(M)';     %系统噪声服从高斯分布
end
for n=1:N
   z(1:M,n)=swerling2pu(M)';   %地物杂波服从swerling2型分布
end
s=v+g+y+z;
%s=v+g;
[m,n]=size(s);
x=zeros(1,m*n);
q=zeros(1,m*n);
x=s(:)';
q=g(:)';
p=x;
i=0:length(p)-1;
subplot(2,1,1);
plot(i,x),title('目标回波信号');          %目标回波信号x
subplot(2,1,2); 
l=canshu(x);
b=x-l(1);
plotpu(b),title('目标回波频谱');         
