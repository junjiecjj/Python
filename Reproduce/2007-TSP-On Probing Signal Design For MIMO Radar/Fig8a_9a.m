clc;
clear all;
close all;

addpath('./functions');

rng(42); 


%% Minimum Sidelobe Beampattern Design. Eq.(32)

%% 参数设置
M = 10;                 % 天线数
c = 1;                  % 总发射功率
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
R_float1 = MinimumSidelobeBeampatternDesignFloatSidelobe(c, M, theta0, theta1, theta2, Omega);

%% 计算并绘制波束图
theta_plot = -90:0.1:90;
P_fixed = zeros(size(theta_plot));
P_float = zeros(size(theta_plot));
P_float1 = zeros(size(theta_plot));
for i = 1:length(theta_plot)
    a_theta = a(theta_plot(i));
    P_fixed(i) = real(a_theta' * R_fixed * a_theta);
    P_float(i) = real(a_theta' * R_float * a_theta);
    P_float1(i) = real(a_theta' * R_float1 * a_theta);
end

% 归一化（dB 显示）
P_fixed_dB = 10*log10(P_fixed / max(P_fixed));
P_float_dB = 10*log10(P_float / max(P_float));
P_float_dB1 = 10*log10(P_float1 / max(P_float1));

figure(1);
plot(theta_plot, P_fixed_dB, 'b-', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_float_dB, 'r--', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_float_dB1, 'k--', 'LineWidth', 1.5); hold on;
xlabel('\theta (degrees)');
ylabel('Normalized Power (dB)');
legend('Fixed elemental power', 'Elemental power 80%~120%', 'width 80%~120%');
title('Minimum Sidelobe Beampattern');
grid on; xlim([-90,90]); ylim([-50,0]);

% 标注主瓣 3dB 宽度
xline(theta1, 'k:', 'LineWidth', 1);
xline(theta2, 'k:', 'LineWidth', 1);
text(theta1-2, -5, '-10°', 'HorizontalAlignment', 'right');
text(theta2+2, -5, '10°', 'HorizontalAlignment', 'left');

% 标注旁瓣峰值电平（固定功率版本）
[peak_vals, peak_locs] = findpeaks(P_fixed_dB, theta_plot, 'MinPeakHeight', -50, 'MinPeakDistance', 5);
% 排除主瓣区域附近的峰值（如靠近 -20° 和 20°）
mask = (peak_locs < -20) | (peak_locs > 20);
peak_vals = peak_vals(mask);
peak_locs = peak_locs(mask);
if ~isempty(peak_vals)
    max_sidelobe = max(peak_vals);
    fprintf('固定功率版本：最高旁瓣电平 = %.2f dB\n', max_sidelobe);
    plot(peak_locs, peak_vals, 'bo', 'MarkerSize', 5); hold on;
end

% 标注旁瓣峰值电平（浮动功率版本）
[peak_vals_f, peak_locs_f] = findpeaks(P_float_dB, theta_plot, 'MinPeakHeight', -50, 'MinPeakDistance', 5);
mask_f = (peak_locs_f < -20) | (peak_locs_f > 20);
peak_vals_f = peak_vals_f(mask_f);
peak_locs_f = peak_locs_f(mask_f);
if ~isempty(peak_vals_f)
    max_sidelobe_f = max(peak_vals_f);
    fprintf('浮动功率版本：最高旁瓣电平 = %.2f dB\n', max_sidelobe_f);
    plot(peak_locs_f, peak_vals_f, 'ro', 'MarkerSize', 5); hold on;
end

% 计算最高旁瓣电平（排除主瓣区域）
[peaks_val_sidelob, locs_val_sidelob] = findpeaks(P_float_dB1, theta_plot, 'MinPeakHeight', -50, 'MinPeakDistance', 5);
mask_side = (locs_val_sidelob < -20) | (locs_val_sidelob > 20);
peaks_val_sidelob = peaks_val_sidelob(mask_side);
locs_val_sidelob = locs_val_sidelob(mask_side);
if ~isempty(peaks_val_sidelob)
    max_sidelobe = max(peaks_val_sidelob);
    fprintf('浮动旁瓣版本：最高旁瓣电平 = %.2f dB\n', max_sidelobe);
    plot(locs_val_sidelob, peaks_val_sidelob, 'ko', 'MarkerSize', 5);
end
ylim([-25, 0]);
hold off;


