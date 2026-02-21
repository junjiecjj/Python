clear all; close all; clc;

%% Parameters
M = 41;                         % Number of sensors (ULA)
N = 200;                        % Number of snapshots
theta_true = 30;               % True DoAs (degrees)
D = length(theta_true);        % Number of sources
mc_runs = 500;                 % Monte Carlo runs
d_lamda = 0.5;                 % Element spacing (wavelengths)
SNR_dB_range = -15:1:15;       % SNR range (dB)

%% Preallocate Results
rmse_music = zeros(1, length(SNR_dB_range));     % MUSIC RMSE
rmse_nystrom = zeros(1, length(SNR_dB_range));   % Nyström RMSE
crb_theoretical = zeros(1, length(SNR_dB_range));% CRB

%% Loop over SNR values
for snr_idx = 1:length(SNR_dB_range)
    SNR_dB = SNR_dB_range(snr_idx);
    noise_pow = 10^(-SNR_dB/10);
    
    errors_music = zeros(1, mc_runs);
    errors_nystrom = zeros(1, mc_runs);
    
    for mc = 1:mc_runs
        %% Signal Model
        theta_true_rad = deg2rad(theta_true);
        A = exp(1j * 2 * pi * d_lamda * (0:M-1)' * sin(theta_true_rad));
        S = (randn(D, N) + 1j * randn(D, N)) / sqrt(2);
        N_matrix = sqrt(noise_pow/2) * (randn(M, N) + 1j * randn(M, N));
        X = A * S + N_matrix;

        %% -------- Standard MUSIC --------
        R_hat = (X * X') / N;
        [U, ~] = eig(R_hat);
        U_n = U(:, 1:end-D); % Noise subspace

        theta_scan = linspace(-40, 40, 1000);
        P_music = zeros(size(theta_scan));

        for k = 1:length(theta_scan)
            a_theta = exp(1j * 2 * pi * d_lamda * (0:M-1)' * sin(deg2rad(theta_scan(k))));
            P_music(k) = 1 / (a_theta' * (U_n * U_n') * a_theta);
        end

        [~, peak_locs] = findpeaks(abs(P_music), 'SortStr', 'descend', 'NPeaks', D);
        theta_est = sort(theta_scan(peak_locs(1:D)));

        errors_music(mc) = norm(theta_est - theta_true, 2);

        %% -------- Nyström MUSIC --------
        Na = 20; % Number of randomly selected sensors (Na < M)
        indices = sort(randperm(M, Na)); % Random sensor subset
        Y = X(indices, :);               % Selected sensors’ output
        A_sub = A(indices, :);           % Sub-array steering matrix

        R_yy = (Y * Y') / N;
        R_xy = (X * Y') / N;

        [U_y, D_y] = eig(R_yy);
        [~, idx] = sort(diag(D_y), 'descend');
        U_y = U_y(:, idx);
        D_y = D_y(idx, idx);

        lambda_y = D_y(1:D, 1:D);
        U_ns = zeros(M, D);
        for i = 1:D
            U_ns(:, i) = (1 / lambda_y(i,i)) * R_xy * U_y(:, i);
        end

        P_nystrom = zeros(size(theta_scan));
        for k = 1:length(theta_scan)
            a_theta = exp(1j * 2 * pi * d_lamda * (0:M-1)' * sin(deg2rad(theta_scan(k))));
            P_nystrom(k) = 1 / (a_theta' * (eye(M) - U_ns * U_ns') * a_theta);
        end

        [~, peak_locs_ny] = findpeaks(abs(P_nystrom), 'SortStr', 'descend', 'NPeaks', D);
        theta_est_ny = sort(theta_scan(peak_locs_ny(1:D)));

        errors_nystrom(mc) = norm(theta_est_ny - theta_true, 2);
    end

    %% Store RMSE
    rmse_music(snr_idx) = sqrt(mean(errors_music.^2));
    rmse_nystrom(snr_idx) = sqrt(mean(errors_nystrom.^2));

    %% Theoretical CRB
    SNR = 10^(SNR_dB / 10);
    A_deriv = zeros(M, D);
    for i = 1:D
        A_deriv(:, i) = 1j * 2 * pi * d_lamda * cos(theta_true_rad(i)) * (0:M-1)' .* A(:, i);
    end

    FIM = zeros(D, D);
    for i = 1:D
        for j = 1:D
            FIM(i, j) = 2 * SNR * N * real(trace((A_deriv(:, i)' * (eye(M) - A * pinv(A' * A) * A') * A_deriv(:, j))));
        end
    end

    crb_matrix = inv(FIM);
    crb_theoretical(snr_idx) = sqrt(trace(crb_matrix)) * (180/pi); % in degrees
end

%% Plot RMSE vs. SNR
figure;
semilogy(SNR_dB_range, rmse_music, 'o-', 'LineWidth', 2, 'DisplayName', 'MUSIC (Joint RMSE)');
hold on;
semilogy(SNR_dB_range, rmse_nystrom, 's-', 'LineWidth', 2, 'DisplayName', 'Nyström MUSIC');
semilogy(SNR_dB_range, crb_theoretical, '--', 'LineWidth', 2, 'DisplayName', 'Joint CRB');
grid on;
xlabel('SNR (dB)');
ylabel('RMSE (degrees)');
title('Joint RMSE (L2-norm) vs. SNR for MUSIC and Nyström MUSIC');
legend('Location', 'NorthEast');

