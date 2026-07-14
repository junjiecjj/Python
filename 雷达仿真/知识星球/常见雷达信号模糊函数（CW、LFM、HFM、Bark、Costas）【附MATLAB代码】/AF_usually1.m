clc;
clear;
close all;
% 用 MATLAB 自带的 ambgfun 比较
%% ============================================================
% 基本参数
% =============================================================
fs = 2e6;
T = 130e-6;
N = round(T * fs);
t = ((0:N-1).' - (N-1) / 2) / fs;

PRF = 1 / T;
B = 200e3;
K = B / T;

%% ============================================================
% 1. 矩形脉冲
% =============================================================
s_rect = ones(N, 1);
s_rect = s_rect / norm(s_rect);

%% ============================================================
% 2. LFM信号
% =============================================================
s_lfm = exp(1j * pi * K * t.^2);
s_lfm = s_lfm / norm(s_lfm);

%% ============================================================
% 3. Barker-13相位编码信号
% =============================================================
barker_code = [1; 1; 1; 1; 1; -1; -1; 1; 1; -1; 1; -1; 1];

samples_per_chip = floor(N / length(barker_code));
s_barker = repelem(barker_code, samples_per_chip);

if length(s_barker) < N
    s_barker = [s_barker; zeros(N - length(s_barker), 1)];
else
    s_barker = s_barker(1:N);
end

s_barker = s_barker / norm(s_barker);

%% ============================================================
% 4. MATLAB自带函数计算二维模糊函数
% =============================================================
[AF_rect, delay_rect, doppler_rect] = ambgfun(s_rect, fs, PRF);
[AF_lfm, delay_lfm, doppler_lfm] = ambgfun(s_lfm, fs, PRF);
[AF_barker, delay_barker, doppler_barker] = ambgfun(s_barker, fs, PRF);

%% ============================================================
% 5. 矩形脉冲二维模糊函数
% =============================================================
figure(1);
mesh(delay_rect * 1e6, doppler_rect / 1e3, AF_rect);
xlabel('Delay (\mus)');
ylabel('Doppler frequency (kHz)');
zlabel('Normalized magnitude');
zlim([0, 1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 6. LFM二维模糊函数
% =============================================================
figure(2);
mesh(delay_lfm * 1e6, doppler_lfm / 1e3, AF_lfm);
xlabel('Delay (\mus)');
ylabel('Doppler frequency (kHz)');
zlabel('Normalized magnitude');
zlim([0, 1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 7. Barker-13二维模糊函数
% =============================================================
figure(3);
mesh(delay_barker * 1e6, doppler_barker / 1e3, AF_barker);
xlabel('Delay (\mus)');
ylabel('Doppler frequency (kHz)');
zlabel('Normalized magnitude');
zlim([0, 1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 8. 零多普勒切面
% =============================================================
[AF_rect_delay, delay_rect_cut] = ambgfun(s_rect, fs, PRF, "Cut", "Delay", "CutValue", 0);
[AF_lfm_delay, delay_lfm_cut] = ambgfun(s_lfm, fs, PRF, "Cut", "Delay", "CutValue", 0);
[AF_barker_delay, delay_barker_cut] = ambgfun(s_barker, fs, PRF, "Cut", "Delay", "CutValue", 0);

figure(4);
plot(delay_rect_cut * 1e6, 20 * log10(AF_rect_delay + eps), 'LineWidth', 1.5);
hold on;
plot(delay_lfm_cut * 1e6, 20 * log10(AF_lfm_delay + eps), 'LineWidth', 1.5);
plot(delay_barker_cut * 1e6, 20 * log10(AF_barker_delay + eps), 'LineWidth', 1.5);
grid on;
xlabel('Delay (\mus)');
ylabel('Normalized magnitude (dB)');
legend('Rectangular pulse', 'LFM', 'Barker-13', 'Location', 'best');
ylim([-60, 3]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 9. 零时延多普勒切面
% =============================================================
[AF_rect_doppler, doppler_rect_cut] = ambgfun(s_rect, fs, PRF, "Cut", "Doppler", "CutValue", 0);
[AF_lfm_doppler, doppler_lfm_cut] = ambgfun(s_lfm, fs, PRF, "Cut", "Doppler", "CutValue", 0);
[AF_barker_doppler, doppler_barker_cut] = ambgfun(s_barker, fs, PRF, "Cut", "Doppler", "CutValue", 0);

figure(5);
plot(doppler_rect_cut / 1e3, 20 * log10(AF_rect_doppler + eps), 'LineWidth', 1.5);
hold on;
plot(doppler_lfm_cut / 1e3, 20 * log10(AF_lfm_doppler + eps), 'LineWidth', 1.5);
plot(doppler_barker_cut / 1e3, 20 * log10(AF_barker_doppler + eps), 'LineWidth', 1.5);
grid on;
xlabel('Doppler frequency (kHz)');
ylabel('Normalized magnitude (dB)');
legend('Rectangular pulse', 'LFM', 'Barker-13', 'Location', 'best');
ylim([-60, 3]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);