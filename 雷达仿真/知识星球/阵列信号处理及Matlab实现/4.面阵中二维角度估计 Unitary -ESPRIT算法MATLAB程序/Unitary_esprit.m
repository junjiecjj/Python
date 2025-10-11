% Developed by xiaofei zhang (ÄÏ¾©º½¿Õº½Ìì´óÑ§ µç×Ó¹¤³ÌÏµ ÕÅÐ¡·É£©
% EMAIL:zhangxiaofei@nuaa.edu.cn
clc;
clear all;
close all;

function p=qq(N)
    k=fix(N/2);
    I=eye(k);
    II=fliplr(I);
    if mod(N,2)==0
        p=[I,j*I;II,-j*II]/sqrt(2);
    else
        p=[I,zeros(k,1),j*I;zeros(1,k),sqrt(2),zeros(1,k);II,zeros(k,1),-j*II]/sqrt(2);
    end
end

derad = pi/180;
radeg = 180/pi;
twpi = 2*pi;
kelm = 8;               % 
dd = 0.5;               % 
d=-(kelm-1)/2*dd:dd:(kelm-1)/2*dd;     % 
iwave = 3;              % number of DOA
theta1 = [10 20 30];  
theta2 = [20 25 15];% DOA
snr = 20;              % input SNR (dB)
n = 200;                % 
A0=exp(j*twpi*d.'*(sin(theta1*derad).*cos(theta2*derad)))/sqrt(kelm);
A1=exp(j*twpi*d.'*(sin(theta1*derad).*sin(theta2*derad)))/sqrt(kelm);%%%% direction matrix
S=randn(iwave,n);
X0=[];
for im=1:kelm
      X0=[X0;A0*diag(A1(im,:))*S];
end
X=awgn(X0,snr,'measured');
L=iwave;
J1=eye(kelm-1,kelm);
J2=flipud(fliplr(J1));%(fliplr:×óÓÒ·­×ª£¬flipud:ÉÏÏÂ·­×ª)
Q=qq(kelm);
Y=kron(Q',Q')*X;
Q0=qq(kelm-1);
K1=real(Q0'*J2*Q);
K2=imag(Q0'*J2*Q);
I=eye(kelm);
Ku1=kron(I,K1);%(Kronecker»ý£¬¾ØÕóIÖÐµÄÃ¿¸öÔªËØ¶¼³ËÒÔ¾ØÕóK1)
Ku2=kron(I,K2);
Kv1=kron(K1,I);
Kv2=kron(K2,I);
E=[real(Y),imag(Y)];
Ey=E*E'/n;
[V,D]=eig(Ey);
EVAs =diag(D).';
[EVAs,I0] = sort(EVAs);
EVAs=fliplr(EVAs);
EVs=fliplr(V(:,I0));
Es=EVs(:,1:L);
fiu=pinv(Ku1*Es)*Ku2*Es;
fiv=pinv(Kv1*Es)*Kv2*Es;
F=fiu+j*fiv;
[VV,DD]=eig(F);
EVA = diag(DD).';
u=2*atan(real(EVA))/pi;
v=2*atan(imag(EVA))/pi;
theta10=asin(sqrt(u.^2+v.^2))*radeg
theta20=atan(v./u)*radeg





