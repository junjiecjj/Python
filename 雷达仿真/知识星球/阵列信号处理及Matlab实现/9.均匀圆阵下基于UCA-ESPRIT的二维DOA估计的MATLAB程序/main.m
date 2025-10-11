%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     2D-DOA estimation with UCA-ESPRIT for circular array      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Developed by lingyun xu and xiaofei zhang (南京航空航天大学 电子工程系 张小飞）
% EMAIL:zhangxiaofei@nuaa.edu.cn

clc;
close all;
clear all;

function v=VV(b)
    m=-4:1:4;
    v=exp(j*m*b);
end

function w=WW(a,n)
    for i=0:1:n-1
        w(i+1)=(1/16)*exp(j*2*pi*a*i/n);
        %w=w';
    end
end

N=16;
M=floor(N/4);
m=-M:1:M;
m1=-1:-1:-M;
cv=diag([(j).^(m(1:M+1)),(j).^(m1(1:M))]);
v=[];
for in=1:2*M+1
    v=[v,WW(in-5,N)'];
end
v=4*v;
fe=v*cv';
a1=2*pi*m/(2*M+1);
W1=[];
for in=1:2*M+1
    W1=[W1,VV(a1(in)).'];
end
W1=(1/3)*W1;
Fr=fe*W1;
x=[1,-1,1,-1,1,1,1,1,1];
c0=diag(x);
x2=c0*W1;
%构造信号
snap=500;         %%快拍数
fs=1000;            %%采样频率
t=[0:snap-1]/fs;
M1=16;               %%阵元数
N1=2;                %%目标数
f=30e3;
R=1/(4*sin(pi/M1));
snr=10;
alpha=[10,20];
theta=[15,35];
%%
uu=sin(alpha*pi/180).*exp(j*theta*pi/180);
a1=[0:(M1-1)]';
for ii=1:N1
    rand('state',ii)
    s(ii,:)=exp(j*2*pi*(f*t+0.5*5*2^(ii-1)*t.^2));
end
for in=1:N1
    A11(:,in)=exp(j*2*pi*cos(2*pi*a1/M1-theta(in)*pi/180)*sin(alpha(in)*pi/180));
end

%%%%%%%%%%%%%%%%
S=s;
 X0=A11*S;
  Y=awgn(X0,snr,'measured');
  RR=Y*Y'/snap;
  RRR=Fr'*RR*Fr;
  [EV,D]=eig(real(RRR));
  % 
EVA=diag(D)';
[EVA,I]=sort(EVA);
EVA=fliplr(EVA);
EV=fliplr(EV(:,I));
E=EV(:,1:N1);
C0=[1,-1,1,-1,1,1,1,1,1];
C00=diag(C0);
Z1=C00*W1*E;
Z11=Z1(1:7,:);
Z12=Z1(2:8,:);
Z13=Z1(3:9,:);
E1=[Z11,Z13];
T1=[-3,-2,-1,0,1,2,3];
T=1/pi*diag(T1);
%T11=diag(T1);
%B=inv(E'*E)*E'*T*Z12;
B=pinv(E1)*T*Z12;
B1=B(1:2,:);
[p,w]=eig(B1);
asin(abs(diag(w)))*180/pi
angle(diag(w))*180/pi



