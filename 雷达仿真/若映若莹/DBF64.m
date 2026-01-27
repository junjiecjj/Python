% https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247489760&idx=1&sn=6d6ecae3256be0ee09262e8d3da06d01&chksm=ce751c7999fc2c68d47df6d85ba0e5a512c26bdab35bf8e7084111205b908fdd0d1e36c85d03&mpshare=1&scene=1&srcid=0127Rz0jVOVDcl702YExF10A&sharer_shareinfo=ff5024f5acd73a47f3dd0029305304e1&sharer_shareinfo_first=ff5024f5acd73a47f3dd0029305304e1&exportkey=n_ChQIAhIQEr0VfidjfJzXTHAYl5om9RKfAgIE97dBBAEAAAAAAKtlKyk9YS0AAAAOpnltbLcz9gKNyK89dVj08wieklirvmjK6GRUWZ6BWOzi0ouR1Ybsk8NXBD4zyjig6P6W057hjMLr7mhvt2n%2FQrJVghFufEPEcIRLkOdMKI40FybWDxa9SCCeJVEJvZH%2FGyx6%2BG8nhXuC%2BG8YVmcHbtN7aL4XM%2BlbWQfbdL%2FGx8Vfy1aGo0yWuxUiWVO563OaGmTUm2usyDL43AipJHndb2JDeS0Tj4qmnr%2FZZVxhwWamUWuuHurGEm%2BDHJOQKFAF5NOv0BjkeUqdqF4QKs9%2FN1jI9WztX1z4BJbtq9GJwnqrk87h%2Bhl4Vw9glJcyZ3Yae1CizSVzDih%2ByeyQwrER9HXVgunYdWs4&acctmode=0&pass_ticket=ZjpyS5oWHSIZ4PyPVhfqIvuqnkHCRlUjUshAJXvpK5WQVLtRrjzF4S0ndTEZlvpK&wx_header=0&poc_token=HFnBeGmj-SxHUsqfSbthIKABqkMO0CGpickHDzwy


% 1、目标方位角维20.463度情况下DBF方向图
% 64阵元圆阵单波束DBF仿真（指向目标方位角20.4630度）
clear; clc; close all;
%% ===================== 参数设置 =====================
N = 64;                  % 阵元数
lambda = 1;              % 波长（简化计算设为1）
R_over_lambda = 5;       % 圆阵半径与波长比 R/λ=5
R = R_over_lambda * lambda; % 圆阵半径
k = 2*pi / lambda;       % 波数
theta =90 * pi/180;     % 俯仰角90°（仅考虑xy平面方位角）
phi_target = 20.4630* pi/180; % 目标方位角（题目给定角度）
%% ===================== 阵元位置计算 =====================
phi_n = 2*pi*(0:N-1)/N;  % 第n个阵元的方位角（0到2π均匀分布）
%% ===================== 导向向量与权重计算 =====================
% 目标方向的导向向量
a_target = exp(1j * k * R * sin(theta) * cos(phi_target - phi_n));
w = (a_target);
%% ===================== 扫描角度与波束响应计算 =====================
phi_scan = linspace(-pi, pi, 1024); % 扫描范围：-180°到180°，1024个点
num_scan = length(phi_scan);
F = zeros(1, num_scan);
for i = 1:num_scan
    phi = phi_scan(i);
    % 当前扫描角度的导向向量
    a_scan = exp(1j * k * R * sin(theta) * cos(phi - phi_n));
    % 波束响应：权重与导向向量的内积
    F(i) = (w) * a_scan';
end
%% ===================== 归一化与绘图 =====================
F_abs = abs(F);
F_dB = 20 * log10(F_abs / max(F_abs)); % 归一化到最大值0dB
figure('Color','w');
plot(phi_scan*180/pi, F_dB, 'LineWidth', 1.2);
xlabel('指向角（度）','FontSize',10);
ylabel('增益（dB）','FontSize',10);
title(['64阵元圆阵DBF方向图（目标方位角=',num2str(phi_target*180/pi),'°，R/\lambda=',num2str(R_over_lambda),')'],'FontSize',11);
grid on;
ylim([-120, 0]); 



% 64阵元圆阵距离门波束输出仿真
clear; clc; close all;
%% ===================== 参数设置 =====================
N = 64;                  % 阵元数
lambda = 1;              % 波长（简化计算设为1）
R_over_lambda = 5;       % R/λ=5
R = R_over_lambda * lambda;
k = 2*pi / lambda;       % 波数
theta = 90 * pi/180;     % 俯仰角90°（仅考虑xy平面方位角）
phi_target = 20.4630 * pi/180; % 目标方位角
%% ===================== 阵元位置计算 =====================
phi_n = 2*pi*(0:N-1)/N;  % 第n个阵元的方位角（0到2π均匀分布）
%% ===================== 生成目标回波（单距离门） =====================
% 模拟一个距离门的回波信号（固定时刻的回波，无时间维度）
s = exp(1j * k * R * sin(theta) * cos(phi_target - phi_n));
% 添加少量噪声，模拟实际雷达回波
s = s + 0.1*(randn(1,N) + 1j*randn(1,N));
%% ===================== 生成64个波束的权重向量 =====================
num_beams = 64;          % 波束数，与阵元数相同
beam_phi = 2*pi*(0:num_beams-1)/num_beams; % 64个波束的指向角（0到2π均匀分布）
w_beams = zeros(num_beams, N);
for b = 1:num_beams
    phi_beam = beam_phi(b);
    % 第b个波束的导向向量
    w_beams(b, :) = (exp(1j * k * R * sin(theta) * cos(phi_beam - phi_n)));
end
%% ===================== 计算每个波束的输出幅度 =====================
beam_output = zeros(1, num_beams);
for b = 1:num_beams
    % 波束输出：权重与回波信号的内积
    beam_output(b) = abs(w_beams(b, :) * s');
end
%% ===================== 绘图 =====================
figure('Color','w');
plot(1:num_beams, beam_output/max(beam_output), 'LineWidth', 1.2);
xlabel('波束','FontSize',10);
ylabel('幅度','FontSize',10);
title('圆阵波束形成后有信号的距离门上64个波束输出','FontSize',11);
grid on;


% 64阵元圆阵64波束覆盖360°仿真
clear; clc; close all;
%% ===================== 参数设置 =====================
N = 64;                  % 阵元数
lambda = 1;              % 波长
R_over_lambda = 5;       % R/λ=5
R = R_over_lambda * lambda;
k = 2*pi / lambda;
theta = 90 * pi/180;     % 俯仰角90°
num_beams = 64;          % 波束数，与阵元数相同，覆盖360°
%% ===================== 阵元位置与波束指向计算 =====================
phi_n = 2*pi*(0:N-1)/N;                  % 阵元方位角
beam_phi = linspace(-pi, pi, num_beams); % 64个波束的指向角（-180°到180°）
%% ===================== 扫描角度与多波束响应计算 =====================
phi_scan = linspace(-pi, pi, 1024);
num_scan = length(phi_scan);
F_beams = zeros(num_beams, num_scan);
for b = 1:num_beams
    phi_beam = beam_phi(b);
    % 第b个波束的导向向量
    a_beam = exp(1j * k * R * sin(theta) * cos(phi_beam - phi_n));
    w_beam = (a_beam); % 权重向量
    % 计算该波束在所有扫描角度的响应
    for i = 1:num_scan
        phi = phi_scan(i);
        a_scan = exp(1j * k * R * sin(theta) * cos(phi - phi_n));
        F_beams(b, i) = w_beam * a_scan';
    end
    % 归一化每个波束的响应
%     F_beams(b, :) = 20*log10(abs(F_beams(b, :)) / max(abs(F_beams(b, :))));
end
figure('Color','w');
for i = 1:num_beams
    plot(phi_scan*180/pi, F_beams(i, :),'k');hold on
end
xlabel('指向角（度）','FontSize',10);
ylabel('增益（dB）','FontSize',10);
title('64阵元圆阵部分波束方向图','FontSize',11);
% legend('波束1','波束21','波束41','Location','best');
grid on;
% ylim([-120, 0]);

%% 64阵元圆阵3维数字波束形成仿真（多目标场景）
clear; clc; close all;
%% ===================== 1. 系统参数配置 =====================
N = 64;          % 阵元数
R_over_lambda = 5; % R/λ
lambda = 1;      % 波长（归一化，不影响角度分辨率）
R = R_over_lambda * lambda;
% 阵元位置计算
phi_n = (0:N-1)' * 2*pi/N; % 第n个阵元的方位角
x = R * cos(phi_n);        % x坐标
y = R * sin(phi_n);        % y坐标
pos = [x, y, zeros(N,1)];  % 阵元位置矩阵
%% ===================== 2. 多目标设置 =====================
% 目标1: 方位30°, 俯仰20°
target1_theta = 20 * pi/180;
target1_phi = 30 * pi/180;
% 目标2: 方位-45°, 俯仰10°
target2_theta = 10 * pi/180;
target2_phi = -45 * pi/180;
% 目标3: 方位120°, 俯仰30°
target3_theta = 30 * pi/180;
target3_phi = 120 * pi/180;
% 生成目标回波信号
s1 = steering_vector1(target1_theta, target1_phi, pos, lambda);
s2 = steering_vector1(target2_theta, target2_phi, pos, lambda);
s3 = steering_vector1(target3_theta, target3_phi, pos, lambda);
noise = (randn(N,1) + 1j*randn(N,1))/sqrt(2); % 加性高斯噪声
rx_signal = s1 + s2 + s3 + 0.1*noise; % 接收信号
%% ===================== 3. 3维波束形成计算 =====================
% 角度网格设置
theta_grid = linspace(0, pi/2, 90);  % 俯仰角 0~90°
phi_grid = linspace(-pi, pi, 180);   % 方位角 -180~180°
[THETA, PHI] = meshgrid(theta_grid, phi_grid);
BF_output = zeros(size(THETA));
% 遍历所有角度计算波束输出
for i = 1:length(phi_grid)
    for j = 1:length(theta_grid)
        theta = theta_grid(j);
        phi = phi_grid(i);
        a = steering_vector1(theta, phi, pos, lambda);
        BF_output(i,j) = abs(a' * rx_signal);
    end
end
% 归一化波束输出
BF_output = BF_output / max(BF_output(:));
%% ===================== 4. 3维可视化 =====================
figure(1);
% 子图1：3维波束方向图
surf(rad2deg(PHI), rad2deg(THETA), BF_output);
shading interp;
xlabel('方位角 (°)');
ylabel('俯仰角 (°)');
zlabel('归一化幅度');
title('64阵元圆阵3维数字波束形成（多目标）');
view(120, 30);
% colorbar;
grid on;


function steering_vector1(target1_theta, target1_phi, pos, lambda)

end