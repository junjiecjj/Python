clear; clc; close all
%% 参数
bandWidth = 250e6; %带宽
fc = 24e9; % 载波频率
T_chrip = 420e-6; %  chirp 持续时间
T_idle = 580e-6; % 两个chirp之间的间隔时间
c = physconst('LightSpeed'); %光速
lambda = c/fc;
%% 频域相位差
dis = [6;6+0.1*lambda];
tau = dis*2/c;
axis_t = linspace(0,T_chrip,128);
axis_range = (0:127)*c/(2*bandWidth);
phi = 2*pi *(fc .* tau + (bandWidth/2/T_chrip)*(2*tau*axis_t-tau.^2));
signal = exp(1j*phi);
freq_domain = fft(signal,size(signal,2),2);
figure;
subplot(321);plot(axis_t,real(signal(1,:)));
subplot(323);plot(axis_range,abs(freq_domain(1,:)));
subplot(325);plot(axis_range,angle(freq_domain(1,:)));
subplot(322);plot(axis_t,real(signal(2,:)));
subplot(324);plot(axis_range,abs(freq_domain(2,:)));
subplot(326);plot(axis_range,angle(freq_domain(2,:)));
%% 时域相位差
phi_diff =phi(1,:)-phi(2,:);
ref_diff = ones(size(axis_t))*2*pi*(fc+bandWidth/2)*(tau(1)-tau(2));
max_error = max(abs(phi_diff-ref_diff))/2/pi*100;
fprintf('误差：%d %\n',max_error);

figure;
subplot(311);plot(phi(1,:));title("目标1 中频相位");
subplot(312);plot(phi(2,:));title("目标2 中频相位");
subplot(313);plot(phi_diff,'DisplayName', "调频信号相位差");hold on;plot(ref_diff,'DisplayName', "参考值");hold off;title("相位差");


