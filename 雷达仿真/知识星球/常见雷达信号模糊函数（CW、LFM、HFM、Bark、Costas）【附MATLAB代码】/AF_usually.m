clc;
clear;
close all;
% 自己编写的 ambiguity_function
%% ============================================================
% 基本参数
% =============================================================
fs = 2e6;
dt = 1 / fs;

T = 130e-6;
N = round(T * fs);
t = ((0:N-1).' - (N-1) / 2) * dt;

B = 200e3;
K = B / T;

num_chips = 13;
chip_width = T / num_chips;
samples_per_chip = round(chip_width * fs);

nfft = 2048;
max_lag = N - 1;

%% ============================================================
% 1. 矩形脉冲
% =============================================================
s_rect = ones(N, 1);
s_rect = normalize_energy(s_rect, dt);

%% ============================================================
% 2. LFM信号
% =============================================================
s_lfm = exp(1j * pi * K * t.^2);
s_lfm = normalize_energy(s_lfm, dt);

%% ============================================================
% 3. 高斯脉冲
% =============================================================
sigma_t = T / 8;
s_gaussian = exp(-t.^2 / (4 * sigma_t^2));
s_gaussian = normalize_energy(s_gaussian, dt);

%% ============================================================
% 4. Barker-13相位编码信号
% =============================================================
barker_code = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1].';
s_barker = repelem(barker_code, samples_per_chip);

if length(s_barker) > N
    s_barker = s_barker(1:N);
elseif length(s_barker) < N
    s_barker = [s_barker; zeros(N - length(s_barker), 1)];
end

s_barker = normalize_energy(s_barker, dt);

%% ============================================================
% 5. 数值计算模糊函数
% =============================================================
[chi_rect, tau, fd] = ambiguity_function(s_rect, fs, nfft, max_lag);
[chi_lfm, ~, ~] = ambiguity_function(s_lfm, fs, nfft, max_lag);
[chi_gaussian, ~, ~] = ambiguity_function(s_gaussian, fs, nfft, max_lag);
[chi_barker, ~, ~] = ambiguity_function(s_barker, fs, nfft, max_lag);

chi_rect_abs = abs(chi_rect);
chi_lfm_abs = abs(chi_lfm);
chi_gaussian_abs = abs(chi_gaussian);
chi_barker_abs = abs(chi_barker);

chi_rect_abs = chi_rect_abs / max(chi_rect_abs(:));
chi_lfm_abs = chi_lfm_abs / max(chi_lfm_abs(:));
chi_gaussian_abs = chi_gaussian_abs / max(chi_gaussian_abs(:));
chi_barker_abs = chi_barker_abs / max(chi_barker_abs(:));

tau_us = tau * 1e6;
fd_kHz = fd / 1e3;

%% ============================================================
% 6. 为绘图降低网格密度
% =============================================================
delay_plot_index = 1:4:length(tau);
doppler_plot_index = 1:8:length(fd);

tau_plot = tau_us(delay_plot_index);
fd_plot = fd_kHz(doppler_plot_index);

chi_rect_plot = chi_rect_abs(delay_plot_index, doppler_plot_index);
chi_lfm_plot = chi_lfm_abs(delay_plot_index, doppler_plot_index);
chi_gaussian_plot = chi_gaussian_abs(delay_plot_index, doppler_plot_index);
chi_barker_plot = chi_barker_abs(delay_plot_index, doppler_plot_index);

%% ============================================================
% 7. 矩形脉冲模糊函数
% =============================================================
figure(1);
mesh(fd_plot, tau_plot, chi_rect_plot);
xlabel('Doppler frequency (kHz)');
ylabel('Delay (\mus)');
zlabel('Normalized magnitude');
xlim([-400, 400]);
ylim([-T * 1e6, T * 1e6]);
zlim([0, 1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 8. LFM模糊函数
% =============================================================
figure(2);
mesh(fd_plot, tau_plot, chi_lfm_plot);
xlabel('Doppler frequency (kHz)');
ylabel('Delay (\mus)');
zlabel('Normalized magnitude');
xlim([-400, 400]);
ylim([-T * 1e6, T * 1e6]);
zlim([0, 1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 9. 高斯脉冲模糊函数
% =============================================================
figure(3);
mesh(fd_plot, tau_plot, chi_gaussian_plot);
xlabel('Doppler frequency (kHz)');
ylabel('Delay (\mus)');
zlabel('Normalized magnitude');
xlim([-400, 400]);
ylim([-T * 1e6, T * 1e6]);
zlim([0, 1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 10. Barker-13模糊函数
% =============================================================
figure(4);
mesh(fd_plot, tau_plot, chi_barker_plot);
xlabel('Doppler frequency (kHz)');
ylabel('Delay (\mus)');
zlabel('Normalized magnitude');
xlim([-400, 400]);
ylim([-T * 1e6, T * 1e6]);
zlim([0, 1]);
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 11. 零多普勒时延切面
% =============================================================
[~, zero_doppler_index] = min(abs(fd));

zero_doppler_rect = chi_rect_abs(:, zero_doppler_index);
zero_doppler_lfm = chi_lfm_abs(:, zero_doppler_index);
zero_doppler_gaussian = chi_gaussian_abs(:, zero_doppler_index);
zero_doppler_barker = chi_barker_abs(:, zero_doppler_index);

figure(5);
plot(tau_us, 20 * log10(zero_doppler_rect + 1e-8), 'LineWidth', 1.5);
hold on;
plot(tau_us, 20 * log10(zero_doppler_lfm + 1e-8), 'LineWidth', 1.5);
plot(tau_us, 20 * log10(zero_doppler_gaussian + 1e-8), 'LineWidth', 1.5);
plot(tau_us, 20 * log10(zero_doppler_barker + 1e-8), 'LineWidth', 1.5);
grid on;
xlabel('Delay (\mus)');
ylabel('Normalized magnitude (dB)');
legend('Rectangular pulse', 'LFM', 'Gaussian pulse', 'Barker-13', 'Location', 'best');
xlim([-T * 1e6, T * 1e6]);
ylim([-60, 3]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 12. 零时延多普勒切面
% =============================================================
[~, zero_delay_index] = min(abs(tau));

zero_delay_rect = chi_rect_abs(zero_delay_index, :);
zero_delay_lfm = chi_lfm_abs(zero_delay_index, :);
zero_delay_gaussian = chi_gaussian_abs(zero_delay_index, :);
zero_delay_barker = chi_barker_abs(zero_delay_index, :);

figure(6);
plot(fd_kHz, 20 * log10(zero_delay_rect + 1e-8), 'LineWidth', 1.5);
hold on;
plot(fd_kHz, 20 * log10(zero_delay_lfm + 1e-8), 'LineWidth', 1.5);
plot(fd_kHz, 20 * log10(zero_delay_gaussian + 1e-8), 'LineWidth', 1.5);
plot(fd_kHz, 20 * log10(zero_delay_barker + 1e-8), 'LineWidth', 1.5);
grid on;
xlabel('Doppler frequency (kHz)');
ylabel('Normalized magnitude (dB)');
legend('Rectangular pulse', 'LFM', 'Gaussian pulse', 'Barker-13', 'Location', 'best');
xlim([-400, 400]);
ylim([-60, 3]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 13. LFM理论模糊函数
% =============================================================
[Tau, Fd] = ndgrid(tau, fd);
overlap_length = T - abs(Tau);

chi_lfm_theory = zeros(size(Tau));
valid_index = abs(Tau) < T;

chi_lfm_theory(valid_index) = overlap_length(valid_index) / T .* sinc((Fd(valid_index) - K * Tau(valid_index)) .* overlap_length(valid_index));
chi_lfm_theory = abs(chi_lfm_theory);
chi_lfm_theory = chi_lfm_theory / max(chi_lfm_theory(:));

lfm_error = max(abs(chi_lfm_abs(:) - chi_lfm_theory(:)));

fprintf('============================================================\n');
fprintf('Ambiguity function simulation\n');
fprintf('Pulse duration:              %.3f us\n', T * 1e6);
fprintf('LFM bandwidth:               %.3f kHz\n', B / 1e3);
fprintf('LFM time-bandwidth product:  %.3f\n', B * T);
fprintf('Barker code length:          %d\n', num_chips);
fprintf('Maximum LFM numerical error: %.6e\n', lfm_error);
fprintf('============================================================\n');

%% ============================================================
% 局部函数
% =============================================================
function x = normalize_energy(x, dt)
energy = sum(abs(x).^2) * dt;
x = x / sqrt(energy);
end

function [chi, tau, fd] = ambiguity_function(x, fs, nfft, max_lag)
x = x(:);
N = length(x);
dt = 1 / fs;

max_lag = min(max_lag, N - 1);
lag_vector = (-max_lag:max_lag).';
num_lags = length(lag_vector);

chi = zeros(num_lags, nfft);

for lag_index = 1:num_lags
    lag = lag_vector(lag_index);
    product = zeros(N, 1);

    if lag >= 0
        valid_index = lag + 1:N;
        product(valid_index) = x(valid_index) .* conj(x(valid_index - lag));
    else
        positive_lag = -lag;
        valid_index = 1:N - positive_lag;
        product(valid_index) = x(valid_index) .* conj(x(valid_index + positive_lag));
    end

    chi(lag_index, :) = fftshift(fft(product, nfft)).' * dt;
end

tau = lag_vector * dt;
fd = (-nfft / 2:nfft / 2 - 1) * fs / nfft;
end