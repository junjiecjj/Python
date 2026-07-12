clc;
clear;
close all;
rng(42);

%% ============================================================
% 基本参数
% =============================================================
c = 3e8;
fc = 10e9;
lambda = c / fc;
d = lambda / 2;

M = 3;
N = 4;
K = 3;
Ns = 128;
L = 200;
SNR_dB = 10;

theta_true = [-35, 5, 40];
alpha_target = [1, 0.8 * exp(1j * 0.7), 0.7 * exp(-1j * 0.5)];

p_tx = (0:M-1).' * N * d;
p_rx = (0:N-1).' * d;

theta_grid = -90:0.1:90;

%% ============================================================
% 1. MIMO虚拟阵列构造
% =============================================================
p_virtual_matrix = p_tx + p_rx.';
p_virtual = p_virtual_matrix(:);
p_virtual = sort(p_virtual);
p_virtual_normalized = p_virtual / d;

fprintf('============================================================\n');
fprintf('1. MIMO virtual array\n');
fprintf('Number of Tx elements: %d\n', M);
fprintf('Number of Rx elements: %d\n', N);
fprintf('Number of virtual elements: %d\n', M * N);
fprintf('Virtual positions normalized by d:\n');
disp(p_virtual_normalized.');

figure(1);
plot(p_tx / d, ones(M, 1), 'o', 'MarkerSize', 9, 'LineWidth', 1.5);
hold on;
plot(p_rx / d, zeros(N, 1), 's', 'MarkerSize', 9, 'LineWidth', 1.5);
plot(p_virtual / d, -ones(M * N, 1), '.', 'MarkerSize', 18);
grid on;
xlabel('Position normalized by d');
ylabel('Array type');
yticks([-1, 0, 1]);
yticklabels({'Virtual array', 'Rx array', 'Tx array'});
legend('Tx elements', 'Rx elements', 'Virtual elements', 'Location', 'best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 2. MIMO正交波形设计
% =============================================================
n = (0:Ns-1).';
S = zeros(Ns, M);

for m = 1:M
    S(:, m) = exp(1j * 2 * pi * (m - 1) * n / Ns) / sqrt(Ns);
end

R_waveform = S' * S;
R_waveform_dB = 20 * log10(abs(R_waveform) + 1e-12);

fprintf('============================================================\n');
fprintf('2. Orthogonal waveform design\n');
fprintf('S^H S =\n');
disp(R_waveform);

figure(2);
imagesc(1:M, 1:M, abs(R_waveform));
axis image;
colorbar;
xlabel('Waveform index');
ylabel('Waveform index');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

figure(3);
plot(real(S(:, 1)), 'LineWidth', 1.2);
hold on;
plot(real(S(:, 2)), 'LineWidth', 1.2);
plot(real(S(:, 3)), 'LineWidth', 1.2);
grid on;
xlabel('Sample index');
ylabel('Real part');
legend('Waveform 1', 'Waveform 2', 'Waveform 3', 'Location', 'best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 3. 匹配滤波和距离处理
% =============================================================
range_bins = [40, 75, 105];
max_delay = 130;
Nr = Ns + max_delay;
Y_rx = zeros(Nr, N);

for k = 1:K
    a_tx_k = exp(1j * 2 * pi / lambda * p_tx * sind(theta_true(k)));
    a_rx_k = exp(1j * 2 * pi / lambda * p_rx * sind(theta_true(k)));

    for m = 1:M
        delayed_waveform = zeros(Nr, 1);
        index_start = range_bins(k);
        index_end = index_start + Ns - 1;

        if index_end <= Nr
            delayed_waveform(index_start:index_end) = S(:, m);
        end

        for r = 1:N
            coefficient = alpha_target(k) * a_tx_k(m) * a_rx_k(r);
            Y_rx(:, r) = Y_rx(:, r) + coefficient * delayed_waveform;
        end
    end
end

signal_power = mean(abs(Y_rx(:)).^2);
noise_power = signal_power / 10^(SNR_dB / 10);
noise = sqrt(noise_power / 2) * (randn(size(Y_rx)) + 1j * randn(size(Y_rx)));
Y_rx_noisy = Y_rx + noise;

matched_output = zeros(Nr + Ns - 1, N, M);

for r = 1:N
    for m = 1:M
        matched_output(:, r, m) = conv(Y_rx_noisy(:, r), conj(flipud(S(:, m))));
    end
end

range_profile = squeeze(sum(sum(abs(matched_output).^2, 2), 3));
range_axis = (0:length(range_profile)-1) - (Ns - 1);

fprintf('============================================================\n');
fprintf('3. Matched filtering\n');
fprintf('True delay bins:\n');
disp(range_bins);

figure(4);
plot(range_axis, 10 * log10(range_profile / max(range_profile) + 1e-12), 'LineWidth', 1.5);
grid on;
xlabel('Delay bin');
ylabel('Normalized output power (dB)');
xlim([0, max_delay + 10]);
ylim([-50, 3]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 4. 虚拟阵列MUSIC DOA估计
% =============================================================
A_virtual = zeros(M * N, K);

for k = 1:K
    a_tx_k = exp(1j * 2 * pi / lambda * p_tx * sind(theta_true(k)));
    a_rx_k = exp(1j * 2 * pi / lambda * p_rx * sind(theta_true(k)));
    A_virtual(:, k) = kron(a_tx_k, a_rx_k);
end

target_snapshots = (randn(K, L) + 1j * randn(K, L)) / sqrt(2);
X_virtual_clean = A_virtual * target_snapshots;

signal_power_virtual = mean(abs(X_virtual_clean(:)).^2);
noise_power_virtual = signal_power_virtual / 10^(SNR_dB / 10);
noise_virtual = sqrt(noise_power_virtual / 2) * (randn(M * N, L) + 1j * randn(M * N, L));
X_virtual = X_virtual_clean + noise_virtual;

R_virtual = X_virtual * X_virtual' / L;
R_virtual = (R_virtual + R_virtual') / 2;

[U, D] = eig(R_virtual, 'vector');
[~, index_sort] = sort(real(D), 'descend');
U = U(:, index_sort);
U_noise = U(:, K + 1:end);

P_music = zeros(size(theta_grid));

for index_theta = 1:length(theta_grid)
    theta = theta_grid(index_theta);
    a_tx = exp(1j * 2 * pi / lambda * p_tx * sind(theta));
    a_rx = exp(1j * 2 * pi / lambda * p_rx * sind(theta));
    a_virtual = kron(a_tx, a_rx);
    denominator = real(a_virtual' * U_noise * U_noise' * a_virtual);
    P_music(index_theta) = 1 / max(denominator, 1e-12);
end

P_music_dB = 10 * log10(P_music / max(P_music));
theta_est_music = select_spectrum_peaks(P_music, theta_grid, K, 5);

fprintf('============================================================\n');
fprintf('4. MUSIC DOA estimation\n');
fprintf('True angles:\n');
disp(theta_true);
fprintf('Estimated MUSIC angles:\n');
disp(theta_est_music);

figure(5);
plot(theta_grid, P_music_dB, 'LineWidth', 1.5);
hold on;

for k = 1:K
    xline(theta_true(k), '--', 'LineWidth', 1.2);
end

grid on;
xlabel('Angle (degree)');
ylabel('Normalized MUSIC spectrum (dB)');
xlim([-90, 90]);
ylim([-50, 3]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 5. MIMO-STAP
% =============================================================
P = 8;
PRF = 2000;
target_angle = 10;
target_doppler = 350;
target_amplitude = 0.3;
num_training = 400;
num_clutter = 80;
clutter_power = 1;
thermal_noise_power = 0.1;
diagonal_loading = 1e-2;

space_time_dimension = M * N * P;
training_data = zeros(space_time_dimension, num_training);

for ell = 1:num_training
    clutter_snapshot = zeros(space_time_dimension, 1);

    for q = 1:num_clutter
        clutter_angle = -70 + 140 * rand;
        normalized_doppler = 0.35 * sind(clutter_angle);
        clutter_doppler = normalized_doppler * PRF;

        a_space = virtual_steering(p_tx, p_rx, lambda, clutter_angle);
        b_doppler = exp(1j * 2 * pi * clutter_doppler / PRF * (0:P-1).');
        s_clutter = kron(b_doppler, a_space);

        clutter_coefficient = sqrt(clutter_power / num_clutter / 2) * (randn + 1j * randn);
        clutter_snapshot = clutter_snapshot + clutter_coefficient * s_clutter;
    end

    thermal_noise = sqrt(thermal_noise_power / 2) * (randn(space_time_dimension, 1) + 1j * randn(space_time_dimension, 1));
    training_data(:, ell) = clutter_snapshot + thermal_noise;
end

R_stap = training_data * training_data' / num_training;
R_stap = (R_stap + R_stap') / 2;
R_stap_loaded = R_stap + diagonal_loading * trace(R_stap) / space_time_dimension * eye(space_time_dimension);

a_target_space = virtual_steering(p_tx, p_rx, lambda, target_angle);
b_target = exp(1j * 2 * pi * target_doppler / PRF * (0:P-1).');
s_target = kron(b_target, a_target_space);

w_stap = R_stap_loaded \ s_target;
w_stap = w_stap / (s_target' * w_stap);

test_clutter = zeros(space_time_dimension, 1);

for q = 1:num_clutter
    clutter_angle = -70 + 140 * rand;
    normalized_doppler = 0.35 * sind(clutter_angle);
    clutter_doppler = normalized_doppler * PRF;

    a_space = virtual_steering(p_tx, p_rx, lambda, clutter_angle);
    b_doppler = exp(1j * 2 * pi * clutter_doppler / PRF * (0:P-1).');
    s_clutter = kron(b_doppler, a_space);

    clutter_coefficient = sqrt(clutter_power / num_clutter / 2) * (randn + 1j * randn);
    test_clutter = test_clutter + clutter_coefficient * s_clutter;
end

test_noise = sqrt(thermal_noise_power / 2) * (randn(space_time_dimension, 1) + 1j * randn(space_time_dimension, 1));
test_snapshot_without_target = test_clutter + test_noise;
test_snapshot_with_target = test_snapshot_without_target + target_amplitude * s_target;

output_without_target = w_stap' * test_snapshot_without_target;
output_with_target = w_stap' * test_snapshot_with_target;

fprintf('============================================================\n');
fprintf('5. MIMO-STAP\n');
fprintf('STAP output without target: %.6f\n', abs(output_without_target));
fprintf('STAP output with target: %.6f\n', abs(output_with_target));

theta_stap_grid = -80:2:80;
doppler_grid = -800:40:800;
P_stap = zeros(length(doppler_grid), length(theta_stap_grid));

for index_theta = 1:length(theta_stap_grid)
    a_space = virtual_steering(p_tx, p_rx, lambda, theta_stap_grid(index_theta));

    for index_doppler = 1:length(doppler_grid)
        b_doppler = exp(1j * 2 * pi * doppler_grid(index_doppler) / PRF * (0:P-1).');
        s_scan = kron(b_doppler, a_space);
        P_stap(index_doppler, index_theta) = abs(w_stap' * s_scan)^2;
    end
end

P_stap_dB = 10 * log10(P_stap / max(P_stap(:)) + 1e-12);

figure(6);
imagesc(theta_stap_grid, doppler_grid, P_stap_dB);
axis xy;
colorbar;
xlabel('Angle (degree)');
ylabel('Doppler frequency (Hz)');
caxis([-50, 0]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ============================================================
% 6. 稀疏MIMO阵列与OMP稀疏DOA恢复
% =============================================================
p_tx_sparse = [0, 5, 11].' * d;
p_rx_sparse = [0, 1, 4, 9].' * d;
p_virtual_sparse = p_tx_sparse + p_rx_sparse.';
p_virtual_sparse = p_virtual_sparse(:);

G = length(theta_grid);
Phi = zeros(M * N, G);

for index_theta = 1:G
    Phi(:, index_theta) = exp(1j * 2 * pi / lambda * p_virtual_sparse * sind(theta_grid(index_theta)));
    Phi(:, index_theta) = Phi(:, index_theta) / norm(Phi(:, index_theta));
end

s_sparse = zeros(G, 1);

for k = 1:K
    [~, index_nearest] = min(abs(theta_grid - theta_true(k)));
    s_sparse(index_nearest) = alpha_target(k);
end

y_sparse_clean = Phi * s_sparse;
signal_power_sparse = mean(abs(y_sparse_clean).^2);
noise_power_sparse = signal_power_sparse / 10^(SNR_dB / 10);
noise_sparse = sqrt(noise_power_sparse / 2) * (randn(M * N, 1) + 1j * randn(M * N, 1));
y_sparse = y_sparse_clean + noise_sparse;

s_omp = omp_complex(Phi, y_sparse, K);
P_omp = abs(s_omp).^2;
P_omp_dB = 10 * log10(P_omp / max(P_omp) + 1e-12);
theta_est_omp = select_spectrum_peaks(P_omp, theta_grid, K, 5);

fprintf('============================================================\n');
fprintf('6. Sparse MIMO array and OMP DOA estimation\n');
fprintf('Sparse Tx positions normalized by d:\n');
disp((p_tx_sparse / d).');
fprintf('Sparse Rx positions normalized by d:\n');
disp((p_rx_sparse / d).');
fprintf('Estimated OMP angles:\n');
disp(theta_est_omp);

figure(7);
stem(theta_grid, P_omp_dB, '.', 'LineWidth', 1.0);
hold on;

for k = 1:K
    xline(theta_true(k), '--', 'LineWidth', 1.2);
end

grid on;
xlabel('Angle (degree)');
ylabel('Normalized OMP spectrum (dB)');
xlim([-90, 90]);
ylim([-50, 3]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

figure(8);
plot(p_tx_sparse / d, ones(M, 1), 'o', 'MarkerSize', 9, 'LineWidth', 1.5);
hold on;
plot(p_rx_sparse / d, zeros(N, 1), 's', 'MarkerSize', 9, 'LineWidth', 1.5);
plot(sort(p_virtual_sparse / d), -ones(M * N, 1), '.', 'MarkerSize', 18);
grid on;
xlabel('Position normalized by d');
ylabel('Array type');
yticks([-1, 0, 1]);
yticklabels({'Virtual array', 'Rx array', 'Tx array'});
legend('Sparse Tx elements', 'Sparse Rx elements', 'Virtual elements', 'Location', 'best');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

fprintf('============================================================\n');
fprintf('Simulation completed.\n');

%% ============================================================
% 局部函数
% =============================================================
function a_virtual = virtual_steering(p_tx, p_rx, lambda, theta)
a_tx = exp(1j * 2 * pi / lambda * p_tx * sind(theta));
a_rx = exp(1j * 2 * pi / lambda * p_rx * sind(theta));
a_virtual = kron(a_tx, a_rx);
end

function theta_est = select_spectrum_peaks(spectrum, theta_grid, number_of_peaks, minimum_separation)
spectrum_work = spectrum(:).';
theta_est = zeros(1, number_of_peaks);

for k = 1:number_of_peaks
    [~, index_peak] = max(spectrum_work);
    theta_est(k) = theta_grid(index_peak);

    forbidden = abs(theta_grid - theta_grid(index_peak)) <= minimum_separation;
    spectrum_work(forbidden) = -inf;
end

theta_est = sort(theta_est);
end

function x_hat = omp_complex(Phi, y, sparsity)
number_of_atoms = size(Phi, 2);
support = [];
residual = y;
x_hat = zeros(number_of_atoms, 1);

for iteration = 1:sparsity
    correlation = abs(Phi' * residual);
    correlation(support) = 0;

    [~, new_index] = max(correlation);
    support = [support, new_index];

    Phi_support = Phi(:, support);
    coefficient_support = Phi_support \ y;
    residual = y - Phi_support * coefficient_support;
end

x_hat(support) = coefficient_support;
end