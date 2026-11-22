% This Matlab script can be used to generate Fig. 12b in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18


clear all;
clc;
close all;
rng('shuffle');

colors = {'#0072BD', '#77AC30', '#7E2F8E', '#D95319', '#E67DAF', '#EDB120', '#A21E2D'};
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'function'));
load(fullfile(root,'data','covMat_BLP.mat'), 'para', 'Vcc', 'Vcc_hat', 'R_eta')
%% Basic system parameters
N = 32;              % Number of subcarriers
L = 64;              % Number of OFDM symbols / slow-time snapshots
para.N = N;
para.L = L;
Nt = 16;             % Number of transmit antennas
Nr = 16;             % Number of receive antennas
fc = para.fc;        % Carrier frequency
deltaf = para.deltaf;% Subcarrier spacing
para.Pt = 10^(50/10) / 1e3;

% Communication and optimization settings
alpha_path = 2.7;    % Path-loss exponent
Path_loss = @(d) -30 - alpha_path * 10 * log10(d);
para.PSD_comm = 10^(-160/10) / 1e3;   % Communication noise PSD
sigma2_comm = para.PSD_comm * deltaf; % Communication noise power
ITER = 100;                           % Number of Monte Carlo trials

opts = struct;
opts.maxAO = 30;           % Maximum AO iterations
opts.maxDinkel = 30;       % Maximum Dinkelbach iterations
opts.tolAO_dB = 0.01;      % AO stopping tolerance in dB
opts.tolDinkel_eta = 1e-3; % Dinkelbach stopping tolerance
opts.lambdaRidge = 0;      % Optional ridge regularization
opts.verbose = 0;          % Verbosity flag

f_n = para.fc + (0:para.N-1) * para.deltaf;   % Frequency grid

% Target parameters
tarPara.M = 1;                         % Number of targets
tarPara.rcs = 10.^([5] / 10);          % Target RCS
tarPara.theta = [-10]';                % Target angle (deg)
tarPara.ranges = [41.8]';              % Target range (m)
tarPara.velos = [-31.2]';              % Target radial velocity (m/s)
tarPara.delays = 2 * tarPara.ranges / para.c0;
tarPara.dopplers = 2 * tarPara.velos / para.lambda_wave;
tarPara.alpha_mn = zeros(tarPara.M, para.N);
Gtx = 10^(20/10);
Grx = 10^(20/10);
for m = 1:tarPara.M
    tarPara.alpha_mn(m, :) = sqrt(para.Pt * (para.c0 ./ f_n).^2 * Gtx * Grx ...
        * tarPara.rcs(m) / ((4*pi)^3 * tarPara.ranges(m)^4));
end
sigma2_tar = abs(tarPara.alpha_mn).^2;

% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);

% Build target steering vectors over subcarriers
a_t = cell(N, 1);
b_t = cell(N, 1);
for n = 0:N-1
    a_t{n+1} = a_n(tarPara.theta, n);
    b_t{n+1} = b_n(tarPara.theta, n);
end
%% Simulation settings
Ku_range = [4 8 12];                    % Numbers of communication users
gammadB_comm_range = 0:4:20;            % Communication SINR thresholds in dB
gamma_comm_range = 10.^(gammadB_comm_range / 10);
nKu = numel(Ku_range);
nGm = numel(gamma_comm_range);
SCNR_perf_ave = nan(nKu, nGm);          % Average SCNR with perfect clutter kernel
SCNR_hat_ave = nan(nKu, nGm);           % Average SCNR with estimated clutter kernel
fail_perf = zeros(nKu, nGm);            % Failure rate with perfect clutter kernel
fail_hat = zeros(nKu, nGm);             % Failure rate with estimated clutter kernel
%% Monte Carlo evaluation
for ig = 1:nGm
    gamma_th = gamma_comm_range(ig);
    fprintf('\n=== Comm SINR threshold = %.1f dB ===\n', gammadB_comm_range(ig));

    for ik = 1:nKu
        Ku = Ku_range(ik);
        para.Ku = Ku;
        gamma_mat = gamma_th * ones(N, Ku);

        sum_perf = 0;
        ok_perf = 0;
        bad_perf = 0;

        sum_hat = 0;
        ok_hat = 0;
        bad_hat = 0;

        parfor it = 1:ITER
            % Generate communication channels
            H_comm = (randn(Nt, N, Ku) + 1j * randn(Nt, N, Ku)) / sqrt(2);

            % Apply path loss for each communication user
            distance_Ku = randi([50, 400], 1, Ku);
            PL_Ku = sqrt(10.^(Path_loss(distance_Ku) / 10));
            for k = 1:Ku
                H_comm(:, :, k) = H_comm(:, :, k) * PL_Ku(k);
            end

            local_sum_perf = 0;
            local_ok_perf = 0;
            local_bad_perf = 0;

            local_sum_hat = 0;
            local_ok_hat = 0;
            local_bad_hat = 0;

            % Perfect clutter kernel
            try
                out_perf = solve_BLP_AO(para, sigma2_tar, Vcc, R_eta, a_t, b_t, ...
                    H_comm, gamma_mat, sigma2_comm, opts);

                scnr_perf_eval = compute_SCNR(out_perf.RX, out_perf.u, Vcc, R_eta, sigma2_tar, a_t, b_t);
                if isfinite(scnr_perf_eval) && isreal(scnr_perf_eval) && (scnr_perf_eval > 0)
                    local_sum_perf = local_sum_perf + scnr_perf_eval;
                    local_ok_perf = local_ok_perf + 1;
                else
                    local_bad_perf = local_bad_perf + 1;
                end
            catch
                local_bad_perf = local_bad_perf + 1;
            end

            % Estimated clutter kernel
            try
                out_hat = solve_BLP_AO(para, sigma2_tar, Vcc_hat, R_eta, a_t, b_t, ...
                    H_comm, gamma_mat, sigma2_comm, opts);

                scnr_hat_eval = compute_SCNR(out_hat.RX, out_hat.u, Vcc, R_eta, sigma2_tar, a_t, b_t);
                if isfinite(scnr_hat_eval) && isreal(scnr_hat_eval) && (scnr_hat_eval > 0)
                    local_sum_hat = local_sum_hat + scnr_hat_eval;
                    local_ok_hat = local_ok_hat + 1;
                else
                    local_bad_hat = local_bad_hat + 1;
                end
            catch
                local_bad_hat = local_bad_hat + 1;
            end

            sum_perf = sum_perf + local_sum_perf;
            ok_perf = ok_perf + local_ok_perf;
            bad_perf = bad_perf + local_bad_perf;

            sum_hat = sum_hat + local_sum_hat;
            ok_hat = ok_hat + local_ok_hat;
            bad_hat = bad_hat + local_bad_hat;
        end

        % Average valid SCNR values
        if ok_perf > 0
            SCNR_perf_ave(ik, ig) = sum_perf / ok_perf;
        end
        if ok_hat > 0
            SCNR_hat_ave(ik, ig) = sum_hat / ok_hat;
        end

        fail_perf(ik, ig) = bad_perf / ITER;
        fail_hat(ik, ig) = bad_hat / ITER;

        fprintf(['Ku=%2d | perfect: valid=%3d/%3d, fail=%.1f%%, SCNR=%.2f dB | ' ...
                 'estimated: valid=%3d/%3d, fail=%.1f%%, SCNR=%.2f dB\n'], ...
            Ku, ok_perf, ITER, 100 * fail_perf(ik, ig), 10 * log10(SCNR_perf_ave(ik, ig)), ...
            ok_hat, ITER, 100 * fail_hat(ik, ig), 10 * log10(SCNR_hat_ave(ik, ig)));
    end
end

SCNRdB_perf = 10 * log10(SCNR_perf_ave);
SCNRdB_hat = 10 * log10(SCNR_hat_ave);
%% Plot SCNR versus communication SINR threshold
h_scnr_gamma = figure('Color', 'white');
plot(gammadB_comm_range, SCNRdB_perf(1, :), '-o', 'Color', colors{1}, 'LineWidth', 1.5);
hold on;
plot(gammadB_comm_range, SCNRdB_hat(1, :), '--o', 'Color', colors{1}, 'LineWidth', 1.5);

plot(gammadB_comm_range, SCNRdB_perf(2, :), '-s', 'Color', colors{2}, 'LineWidth', 1.5);
plot(gammadB_comm_range, SCNRdB_hat(2, :), '--s', 'Color', colors{2}, 'LineWidth', 1.5);

plot(gammadB_comm_range, SCNRdB_perf(3, :), '-^', 'Color', colors{3}, 'LineWidth', 1.5);
plot(gammadB_comm_range, SCNRdB_hat(3, :), '--^', 'Color', colors{3}, 'LineWidth', 1.5);
grid on;
xlabel('Communication SINR threshold (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('SCNR (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
legend('$K=4$ (perfect)', '$K=4$ (estimated)', ...
    '$K=8$ (perfect)', '$K=8$ (estimated)', ...
    '$K=12$ (perfect)', '$K=12$ (estimated)', ...
    'Position', [0.129971869961265 0.109226194903966 0.316127688648897 0.266904753049215], ...
    'FontSize', 11, 'FontName', 'Times New Roman', 'Interpreter', 'latex');