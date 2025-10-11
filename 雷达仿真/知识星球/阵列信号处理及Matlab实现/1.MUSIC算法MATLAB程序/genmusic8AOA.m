clear all
close all
clc
derad = pi/180;        % deg -> rad
radeg = 180/pi;
twpi = 2*pi;
kelm = 8;               % 阵列数量
dd = 0.5;               % space
d=0:dd:(kelm-1)*dd;     %
iwave = 4;              % number of DOA
theta = [ -60 -30 30 60];     % 角度
snr = 40;               % input SNR (dB)
n = 500;                 %
A=exp(-j*twpi*d.'*sin(theta*derad));%%%% direction matrix
S=randn(iwave,n);
X=A*S;
X1=awgn(X,snr,'measured');%在信号X中加入高斯白噪声，信噪比SNR，'measured'函数在加入噪声前测定信号强度
Rxx=X1*X1'/n;  
% InvS=inv(Rxx); %%%%
[EV,D]=eig(Rxx);%%%% [V,D]=eig(A)：求矩阵A的全部特征值，构成对角阵D，并求A的特征向量构成V的列向量。
EVA=diag(D)';%返回矩阵D的主对角线上的元素
[EVA,I]=sort(EVA);%其中I是一个大小等于size(EVA)的数组，其每一列是EVAA中列向量的元素相对应的置换位置记号。  EVA=fliplr(EVA);%将矩阵A的列绕垂直轴进行左右翻转，如果A是一个行向量，fliplr(A)将A中元素的顺序进行翻转。如果A是一个列向量，fliplr(A)还等于A。 EV=fliplr(EV(:,I));
EVA=fliplr(EVA);%将矩阵A的列绕垂直轴进行左右翻转，如果A是一个行向量，fliplr(A)将A中元素的顺序进行翻转。如果A是一个列向量，fliplr(A)还等于A。
EV=fliplr(EV(:,I));
% MUSIC
for iang = 1:361
    angle(iang)=(iang-181)/2;
    phim=derad*angle(iang);
    a=exp(-j*twpi*d*sin(phim)).';
    L=iwave;
    En=EV(:,L+1:kelm);
    SP(iang)=(a'*a)/(a'*En*En'*a);
end   %画图
SP=abs(SP);
SPmax=max(SP);
SP=10*log10(SP/SPmax);
h=plot(angle,SP);
set(h,'Linewidth',2)
xlabel('angle (degree)')
ylabel('magnitude (dB)')
axis([-90 90 -60 0])
set(gca, 'XTick',[-90:30:90])
grid on