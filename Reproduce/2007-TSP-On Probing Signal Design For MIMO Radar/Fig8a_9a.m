clc;
clear all;
close all;

addpath('./functions');
rng(42); 

%% Minimum Sidelobe Beampattern Design. Eq.(32)

%% 参数设置
M = 10;                 % 天线数
c = ones(M,1) * 1/M;        % 对角元固定值
% c = rand(M, 1)
theta0 = 0;             % 主瓣中心（度）
theta1 = -10;           % 3dB 左边界
theta2 = 10;            % 3dB 右边界

% 旁瓣区域（离散网格，步长 0.1°）
sidelobe_left  = -90:0.1:-20;
sidelobe_right = 20:0.1:90;
Omega = [sidelobe_left, sidelobe_right];   % 所有旁瓣角度
% 导向矢量函数（半波长间距）
a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

%% 固定阵元功率约束 (R_mm = c/M)
R_fixed = MinimumSidelobeBeampatternDesign(c, M, theta0, theta1, theta2, Omega);

%% 允许阵元功率浮动 (80% ~ 120% of c/M, 总功率仍为 c)
R_float = MinimumSidelobeBeampatternDesignFloatPow(c, M, theta0, theta1, theta2, Omega);
 
%% 放宽3dB宽度（阵元功率固定为 c/M）
R_float1 = MinimumSidelobeBeampatternDesignRelaxBeamwidth(c, M, theta0, theta1, theta2, Omega);

%% 计算并绘制波束图
theta_plot = -90:0.1:90;
P_fixed = zeros(size(theta_plot));
P_float = zeros(size(theta_plot));
P_relax = zeros(size(theta_plot));

for i = 1:length(theta_plot)
    a_theta = a(theta_plot(i));
    P_fixed(i) = real(a_theta' * R_fixed * a_theta);
    P_float(i) = real(a_theta' * R_float * a_theta);
    P_relax(i) = real(a_theta' * R_float1 * a_theta);
end

% 避免数值误差导致 log10 出现负数或零
P_fixed(P_fixed < 0) = 0;
P_float(P_float < 0) = 0;
P_relax(P_relax < 0) = 0;

% 归一化 dB 显示
P_fixed_dB = 10 * log10(P_fixed / max(P_fixed) + eps);
P_float_dB = 10 * log10(P_float / max(P_float) + eps);
P_relax_dB = 10 * log10(P_relax / max(P_relax) + eps);

% 旁瓣区域
side_idx = theta_plot <= -20 | theta_plot >= 20;

% 直接计算最高旁瓣电平
[max_sidelobe_fixed, idx_fixed] = max(P_fixed_dB(side_idx));
[max_sidelobe_float, idx_float] = max(P_float_dB(side_idx));
[max_sidelobe_relax, idx_relax] = max(P_relax_dB(side_idx));

theta_side = theta_plot(side_idx);
theta_sidelobe_fixed = theta_side(idx_fixed);
theta_sidelobe_float = theta_side(idx_float);
theta_sidelobe_relax = theta_side(idx_relax);

fprintf('固定功率版本：最高旁瓣电平 = %.2f dB，位置 = %.2f°\n', max_sidelobe_fixed, theta_sidelobe_fixed);
fprintf('浮动功率版本：最高旁瓣电平 = %.2f dB，位置 = %.2f°\n', max_sidelobe_float, theta_sidelobe_float);
fprintf('放宽3dB宽度版本：最高旁瓣电平 = %.2f dB，位置 = %.2f°\n', max_sidelobe_relax, theta_sidelobe_relax);

%% 画图
figure(1);
plot(theta_plot, P_fixed_dB, 'b-', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_float_dB, 'r--', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_relax_dB, 'k--', 'LineWidth', 1.5);

% 标注最高旁瓣位置
plot(theta_sidelobe_fixed, max_sidelobe_fixed, 'bo', 'MarkerSize', 6, 'LineWidth', 1.2);
plot(theta_sidelobe_float, max_sidelobe_float, 'ro', 'MarkerSize', 6, 'LineWidth', 1.2);
plot(theta_sidelobe_relax, max_sidelobe_relax, 'ko', 'MarkerSize', 6, 'LineWidth', 1.2);

% 标注主瓣 3dB 宽度
xline(theta1, 'k:', 'LineWidth', 1);
xline(theta2, 'k:', 'LineWidth', 1);
text(theta1 - 2, -5, '-10°', 'HorizontalAlignment', 'right');
text(theta2 + 2, -5, '10°', 'HorizontalAlignment', 'left');

% 标注最高旁瓣电平文字
text(theta_sidelobe_fixed, max_sidelobe_fixed + 1, sprintf('%.2f dB', max_sidelobe_fixed), 'Color', 'b', 'HorizontalAlignment', 'center');
text(theta_sidelobe_float, max_sidelobe_float + 1, sprintf('%.2f dB', max_sidelobe_float), 'Color', 'r', 'HorizontalAlignment', 'center');
text(theta_sidelobe_relax, max_sidelobe_relax + 1, sprintf('%.2f dB', max_sidelobe_relax), 'Color', 'k', 'HorizontalAlignment', 'center');

xlabel('\theta (degrees)');
ylabel('Normalized Power (dB)');
legend('Fixed elemental power', 'Elemental power 80%~120%', 'Relaxed 3dB beamwidth', ...
    'Fixed max sidelobe', 'Float max sidelobe', 'Relax max sidelobe', ...
    'Location', 'best');
title('Minimum Sidelobe Beampattern');
grid on;
xlim([-90, 90]);
ylim([-25, 0]);
hold off;
