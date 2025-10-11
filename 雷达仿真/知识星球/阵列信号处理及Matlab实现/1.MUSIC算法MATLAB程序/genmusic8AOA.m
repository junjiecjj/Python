clear all
close all
clc
derad = pi/180;        % deg -> rad
radeg = 180/pi;
twpi = 2*pi;
kelm = 8;               % ��������
dd = 0.5;               % space
d=0:dd:(kelm-1)*dd;     %
iwave = 4;              % number of DOA
theta = [ -60 -30 30 60];     % �Ƕ�
snr = 40;               % input SNR (dB)
n = 500;                 %
A=exp(-j*twpi*d.'*sin(theta*derad));%%%% direction matrix
S=randn(iwave,n);
X=A*S;
X1=awgn(X,snr,'measured');%���ź�X�м����˹�������������SNR��'measured'�����ڼ�������ǰ�ⶨ�ź�ǿ��
Rxx=X1*X1'/n;  
% InvS=inv(Rxx); %%%%
[EV,D]=eig(Rxx);%%%% [V,D]=eig(A)�������A��ȫ������ֵ�����ɶԽ���D������A��������������V����������
EVA=diag(D)';%���ؾ���D�����Խ����ϵ�Ԫ��
[EVA,I]=sort(EVA);%����I��һ����С����size(EVA)�����飬��ÿһ����EVAA����������Ԫ�����Ӧ���û�λ�üǺš�  EVA=fliplr(EVA);%������A�����ƴ�ֱ��������ҷ�ת�����A��һ����������fliplr(A)��A��Ԫ�ص�˳����з�ת�����A��һ����������fliplr(A)������A�� EV=fliplr(EV(:,I));
EVA=fliplr(EVA);%������A�����ƴ�ֱ��������ҷ�ת�����A��һ����������fliplr(A)��A��Ԫ�ص�˳����з�ת�����A��һ����������fliplr(A)������A��
EV=fliplr(EV(:,I));
% MUSIC
for iang = 1:361
    angle(iang)=(iang-181)/2;
    phim=derad*angle(iang);
    a=exp(-j*twpi*d*sin(phim)).';
    L=iwave;
    En=EV(:,L+1:kelm);
    SP(iang)=(a'*a)/(a'*En*En'*a);
end   %��ͼ
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