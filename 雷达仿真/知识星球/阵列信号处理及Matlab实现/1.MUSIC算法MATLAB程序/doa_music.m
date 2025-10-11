% DOA estimation by MUSIC 
% Developed by xiaofei zhang (ÄÏ¾©º½¿Õº½Ìì´óÑ§ µç×Ó¹¤³ÌÏµ ÕÅĞ¡·É£©
% EMAIL:zhangxiaofei@nuaa.edu.cn
clear all
close all
derad = pi/180;        % deg -> rad
radeg = 180/pi;
twpi = 2*pi;
kelm = 8;               % ÕóÁĞÊıÁ¿
dd = 0.5;               % space 
d=0:dd:(kelm-1)*dd;     % 
iwave = 3;              % number of DOA
theta = [10 30 60];     % ½Ç¶È
snr = 10;               % input SNR (dB)
n = 500;                 % 
A=exp(-j*twpi*d.'*sin(theta*derad));%%%% direction matrix
S=randn(iwave,n);
X=A*S;
X1=awgn(X,snr,'measured');
Rxx=X1*X1'/n;
InvS=inv(Rxx); %%%%ÇóÄæ
[EV,D]=eig(Rxx);%%%% (eig:·µ»Ø¾ØÕóµÄÌØÕ÷ÏòÁ¿ºÍÌØÕ÷Öµ)
EVA=diag(D)';
[EVA,I]=sort(EVA);
EVA=fliplr(EVA);%£¨fliplr£º×óÓÒ·­×ª£©
EV=fliplr(EV(:,I));

% MUSIC
for iang = 1:361
        angle(iang)=(iang-181)/2;
        phim=derad*angle(iang);
        a=exp(-j*twpi*d*sin(phim)).';
        L=iwave;    
        Un=EV(:,L+1:kelm);      %ÔëÉù×Ó¿Õ¼ä
        SP(iang)=1/(a'*Un*Un'*a);
end
   
% 
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




