%%two dimensinal DOA estimation using 2D-MUSIC algorith for L-shaped array
% Developed by xiaofei zhang (南京航空航天大学 电子工程系 张小飞）
% EMAIL:zhangxiaofei@nuaa.edu.cn, fei_zxf@163.com

clear all
close all
clc

twpi = 2*pi;
rad = pi/180;
deg = 180/pi;

kelm = 8;
snr  = 10;
iwave = 3;
theta = [10 30 50];
fe = [15 25 35];
n = 100;
dd = 0.5;
d = 0:dd:(kelm-1)*dd;
d1 = dd:dd:(kelm-1)*dd;
Ax = exp(-j*twpi*d.'*(sin(theta*rad).*cos(fe*rad)));
Ay = exp(-j*twpi*d1.'*(sin(theta*rad).*sin(fe*rad)));
A = [Ax;Ay];
S = randn(iwave,n);
X = A*S;
X1 = awgn(X,snr,'measured');
Rxx = X1*X1'/n;
[EV,D] = eig(Rxx);
[EVA,I] = sort(diag(D).');
EV = fliplr(EV(:,I));
Un = EV(:,iwave+1:end);
for ang1 = 1:90
    for ang2 = 1:90
        thet(ang1) = ang1-1;
        phim1 = thet(ang1)*rad;
        f(ang2) = ang2-1;
        phim2 = f(ang2)*rad;
        a1 = exp(-j*twpi*d.'*sin(phim1)*cos(phim2));
        a2 = exp(-j*twpi*d1.'*sin(phim1)*sin(phim2));
        a = [a1;a2];
        SP(ang1,ang2) = 1/(a'*Un*Un'*a);
    end
end
SP=abs(SP);
SPmax=max(max(SP));
SP=SP/SPmax; 
h = mesh(thet,f,SP);
set(h,'Linewidth',3);
xlabel('elevation(degree)');
ylabel('azimuth(degree)');
zlabel('magnitude(dB)');

% view(90,0);


