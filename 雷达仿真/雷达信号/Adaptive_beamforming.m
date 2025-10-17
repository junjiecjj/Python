clear;
clc;
close all;
% https://mp.weixin.qq.com/s?__biz=Mzk5MDU0NzkwNw==&mid=2247484037&idx=1&sn=5f93f7eb7e129dc9691e2f49bd1b1057&chksm=c4b17f2e00031f2e0f6d5673e8599227e1dccc15fcb94fd7616d4b5650fb5770b25317dd1970&mpshare=1&scene=1&srcid=0722azbNRXBBkfTLWi6VIEro&sharer_shareinfo=aea65b0f1347803ade1c36b892860373&sharer_shareinfo_first=aea65b0f1347803ade1c36b892860373&exportkey=n_ChQIAhIQ361krEeghmIqxJCFXD1HRhKfAgIE97dBBAEAAAAAAGEfBOVPjxMAAAAOpnltbLcz9gKNyK89dVj0c8KtdB2pnZwhGVPrVHvkeytuxBoI6Vc6Nf%2FG%2FToSj5p76aP2G8XGZnY%2B1zO11qJXRX7UbWxObDRrG%2BNCVYhu74XmvCAgsxVnJC0ppwd%2FpQq5sepcCsAxd%2BEag5FC9CrUNqzEynZLvyl8xMjkoOb9b0GacMQY04%2B5irHed64eCpO4ylidkRJ%2F%2Bgh%2FHTb09I1wvmbVVHkLzp1Xg6j%2BvriTP%2BCzNfPgJQHXnR50FiekHuRHkgE1faMZVnq2UJh7E%2F3dLpxM5Yc3Tah9I4DSDKfgToopEYfL6P9f9iCQ20UHdhgMl9gVUvSxHqP22QjWGLb4Cm8WU%2BDflmJm&acctmode=0&pass_ticket=plRn8aPLvJtt%2BA0G7dQxXziufeRX5Xej4t59RjTBfUcg364MtrJx9rkjkO3mOZoI&wx_header=0#rd



%% 信号数据

M = 40;
c = 3e8;
N=300;
f = 4.5*1e6;
lambda = c./f;
SNR = [0, 30, 40];
R_circ = 175;
thetaK = [50, 20, 40]/180*pi;
phiK = [120, 240, 140]/180*pi;
As = 10.^(SNR./20);
A = [exp(1i*2*pi/lambda*R_circ*sin(thetaK(1))*cos(phiK(1)-2*pi*(0:M-1)/M));exp(1i*2*pi/lambda*R_circ*sin(thetaK(2))*cos(phiK(2)-2*pi*(0:M-1)/M));exp(1i*2*pi/lambda*R_circ*sin(thetaK(3))*cos(phiK(3)-2*pi*(0:M-1)/M))].';
signal=randn(length(thetaK),N);
for i=1:1:length(SNR)
    signal(i, :) = signal(i, :)*As(i);
end
noise = (randn(M, N)+1i*randn(M, N))/sqrt(2);
xn = A*signal+noise;


%%  MVDR
% R = xn*xn'/N;
% expect_n=1;
% noise_n=2;
% noise_nn=3;
% A_expect = A(:, expect_n);
% w0 = inv(R)*A_expect/(A_expect'*inv(R)*A_expect);
% 
% dis = 1;
% theta_scan = (0:dis:90-1)/180*pi;
% phi_scan = (0:dis:360-1)/180*pi;
% 
% 
% F = zeros(length(theta_scan), length(phi_scan));
% for i=1:length(theta_scan)
%     for j=1:length(phi_scan)
%         a = exp(1i*2*pi/lambda(expect_n)*R_circ*sin(theta_scan(i))*cos(phi_scan(j)-2*pi*(0:M-1)/M).');
%         F(i, j) = w0'*a;
%     end
% end
% F_abs = 20*log10(abs(F));

%% LCMV
R = xn*xn'/N;
dis = 1;
theta_scan = (0:dis:90-1)/180*pi;
phi_scan = (0:dis:360-1)/180*pi;
C_matrix = A;
matrix_f = [1,0,0].'; %约束向量
F = zeros(length(theta_scan), length(phi_scan));

w0 = inv(R)*C_matrix/(C_matrix'*inv(R)*C_matrix)*matrix_f;

for i=1:length(theta_scan)
    for j=1:length(phi_scan)
        a = exp(1i*2*pi/lambda(1)*R_circ* sin(theta_scan(i))*cos(phi_scan(j)-2*pi*(0:M-1)/M).');
        F(i, j) = w0'*a;
    end
end
F_abs = 20*log10(abs(F));


%% RAB
% R = xn*xn'/N;
% dis = 1;% 扫描间隔
% theta_scan = (0:dis:90-1)/180*pi;
% phi_scan = (0:dis:360-1)/180*pi;
% 
% [E_all, lambda_all] = eig(R) ;
% Es = E_all(:, end:-1:end-length(thetaK)+1);
% C_matrix = A(:, 1);
% F = zeros(length(theta_scan), length(phi_scan));
% C_matrix = Es*Es'*C_matrix;
% w0 = inv(R)*C_matrix/(C_matrix'*inv(R)*C_matrix);
% 
% for i=1:length(theta_scan)
%     for j=1:length(phi_scan)
%         a = exp(1i*2*pi/lambda(1)*R_circ*sin(theta_scan(i))*cos(phi_scan(j)-2*pi*(0:M-1)/M).');
%         F(i, j) = w0'*a;
%     end
% end
% 
% F_abs = 20*log10(abs(F));


%% 画图
figure(1)
mesh(phi_scan*180/pi,theta_scan*180/pi,F_abs)
colorbar;
xlabel('方位角 \phi °');
ylabel('俯仰角 \theta ° ');
zlabel('波束增益 (dB)');
title(['M=',num2str(M),'阵元 环阵CBF三维视图 来波f=',num2str(f/1e6),'MHz']);
set(gca,'fontsize',12);
axis([0 360 0 90 min(min(F_abs))-20 max(max(F_abs))+20])
set(gca,'XTick',0:60:360);
set(gca,'YTick',0:15:90);
set(gca,'ZTick',(round(min(min(F_abs))/10)*10-10):20:20);

expect_n=1;
noise_n=2;
noise_nn=3;
figure(2)
subplot(3,2,1)
plot(theta_scan*180/pi,F_abs(:, phi_scan==phiK(expect_n)),  'HandleVisibility', 'off')
hold on;
plot([thetaK(expect_n)/pi*180, thetaK(expect_n)/pi*180], [min(F_abs(:,phi_scan==phiK(expect_n))),40])
hold off;
title('期望方向俯仰角切片')
axis([0 90 min(F_abs(:,phi_scan==phiK(expect_n))) max(F_abs(:,phi_scan==phiK(expect_n)))+20])
set(gca,'XTick',0:15:90);
legend(['\theta=',num2str(thetaK(expect_n)/pi*180),'°']);

subplot(3,2,2)
plot(phi_scan*180/pi,F_abs(theta_scan==thetaK(expect_n), :),  'HandleVisibility', 'off')
hold on;
plot([phiK(expect_n)/pi*180, phiK(expect_n)/pi*180], [min(F_abs(theta_scan==thetaK(expect_n), :)),40])
hold off;
title('期望方向方位角切片')
axis([0 360 min(F_abs(theta_scan==thetaK(noise_n), :)) max(F_abs(theta_scan==thetaK(noise_n), :))+20])
set(gca,'XTick',0:60:360);
legend(['\phi=',num2str(phiK(expect_n)/pi*180),'°']);


subplot(3,2,3)
plot(theta_scan*180/pi,F_abs(:,phi_scan==phiK(noise_n)), 'HandleVisibility', 'off')
hold on;
plot([thetaK(noise_n)/pi*180, thetaK(noise_n)/pi*180], [min(F_abs(:,phi_scan==phiK(noise_n))),40])
hold off;
title('干扰方向1俯仰角切片')
axis([0 90 min(F_abs(:,phi_scan==phiK(noise_nn))) max(F_abs(:,phi_scan==phiK(noise_nn)))+20])
set(gca,'XTick',0:15:90);
legend(['\theta=',num2str(thetaK(noise_n)/pi*180),'°']);

subplot(3,2,4)
plot(phi_scan*180/pi,F_abs(theta_scan==thetaK(noise_n), :), 'HandleVisibility', 'off')
hold on;
plot([phiK(noise_n)/pi*180, phiK(noise_n)/pi*180], [min(F_abs(theta_scan==thetaK(noise_n), :)),40])
hold off;
title('干扰方向1方位角切片')
axis([0 360 min(F_abs(theta_scan==thetaK(noise_n), :)) max(F_abs(theta_scan==thetaK(noise_n), :))+20])
set(gca,'XTick',0:60:360);
legend(['\phi=',num2str(phiK(noise_n)/pi*180),'°']);

subplot(3,2,5)
plot(theta_scan*180/pi,F_abs(:,phi_scan==phiK(noise_nn)), 'HandleVisibility', 'off')
hold on;
plot([thetaK(noise_nn)/pi*180, thetaK(noise_nn)/pi*180], [min(F_abs(:,phi_scan==phiK(noise_nn))),40])
hold off;
title('干扰方向2俯仰角切片')
axis([0 90 min(F_abs(:,phi_scan==phiK(noise_nn))) max(F_abs(:,phi_scan==phiK(noise_nn)))+20])
set(gca,'XTick',0:15:90);
legend(['\theta=',num2str(thetaK(noise_nn)/pi*180),'°']);

subplot(3,2,6)
plot(phi_scan*180/pi,F_abs(theta_scan==thetaK(noise_nn), :), 'HandleVisibility', 'off')
hold on;
plot([phiK(noise_nn)/pi*180, phiK(noise_nn)/pi*180], [min(F_abs(theta_scan==thetaK(noise_nn), :)),40])
hold off;
title('干扰方向2方位角切片')
axis([0 360 min(F_abs(theta_scan==thetaK(noise_nn), :)) max(F_abs(theta_scan==thetaK(noise_nn), :))+20])
set(gca,'XTick',0:60:360);
legend(['\phi=',num2str(phiK(noise_nn)/pi*180),'°']);
sgtitle(['M=',num2str(M),'阵元 环阵CBF二维切片 来波f=', num2str(f(1)/1e6),'MHz'])
