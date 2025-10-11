%宽带信号DOA估计算法：非相干信号子空间（ISM）算法 
% Developed by xiaofei zhang (南京航空航天大学 电子工程系 张小飞）
% EMAIL:zhangxiaofei@nuaa.edu.cn


clc 
clear all 
close all 
M=12;                                      %阵元数 
N=200;                                    %快拍数 
ts=0.01;                                  %时域采样间隔 
f0=100;                                   %入射信号中心频率  
f1=80;                                    %入射信号最低频率 
f2=120;                                   %入射信号最高频率 
c=1500;                                   %声速 
lambda=c/f0;                              %波长 
d=lambda/2;                               %阵元间距 
SNR=15;                                   %信噪比 
b=pi/180;   
theat1=30*b;                               %入射信号波束角1 
theat2=0*b;                                %入射信号波束角2 
n=ts:ts:N*ts;                             
theat=[theat1 theat2]'; 
 
%%%%%%%%%%%%%%%% produce signal %%%%%%%%%%%%%%%% 
 
s1=chirp(n,80,1,120);                     %生成线性调频信号1； 
sa=fft(s1,2048);                          %进行FFT变换 
%figure, %specgram(s1,256,1E3,256,250);   %频谱图
s2=chirp(n+0.100,80,1,120);                %生成线性调频信号2 
sb=fft(s2,2048);                           %进行FFT变换 
  
% %%%%%%%%%%%%%%%%%%%%% ISM算法 %%%%%%%%%%%%%%%%%% 
P=1:2; 
a=zeros(M,2); 
sump=zeros(1,181); 
for i=1:N
    f=80+(i-1)*1.0; 
    s=[sa(i) sb(i)]'; 
    for m=1:M 
        a(m,P)=exp(-j*2*pi*f*d/c*sin(theat(P))*(m-1))'; 
    end 
    R=a*(s*s')*a'; 
    [em,zm]=eig(R); 
    [zm1,pos1]=max(zm); 
    for l=1:2 
        [zm2,pos2]=max(zm1); 
        zm1(:,pos2)=[]; 
        em(:,pos2)=[]; 
    end 
    k=1; 
    for ii=-90:1:90 
        arfa=sin(ii*b)*d/c; 
        for iii=1:M 
            tao(1,iii)=(iii-1)*arfa; 
        end 
        A=[exp(-j*2*pi*f*tao)]'; 
        p(k)=A'*em*em'*A; 
        k=k+1; 
    end 
    sump=sump+abs(p); 
end 
pmusic=1/33*sump; 
pm=1./pmusic; 
thetaesti=-90:1:90; 
plot(thetaesti,20*log(abs(pm))); 
xlabel('入射角/度'); 
ylabel('空间谱/dB'); 
grid on
