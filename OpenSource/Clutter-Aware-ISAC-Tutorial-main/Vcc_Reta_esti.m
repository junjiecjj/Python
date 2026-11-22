% This Matlab script is used to generate the clutter-related covariance
% matrices for the BLP-based design in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18

clear;
clc;
close all;
rng('shuffle');

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'function'));
load(fullfile(root,'data','parameters_basic.mat'), 'para', 'tarPara', 'hotSourcePara', 'cfg')
%% Basic parameters
N = 32;                     % Number of subcarriers used in this script
para.N = N;
L = 512;                    % Number of slow-time snapshots
para.L = L;
Nt = para.Nt;               % Number of transmit antennas
Nr = para.Nr;               % Number of receive antennas
Nt_hs = hotSourcePara.Nt_hs;% Number of hot-source antennas
Ku = para.Ku;               % Number of communication users
Ns = para.Ns;               % Number of transmit streams
Pt = para.Pt;               % Total transmit power
deltaf = para.deltaf;       % Subcarrier spacing
fc = para.fc;               % Carrier frequency
Tsym = para.Tsym;           % OFDM symbol duration
c0 = para.c0;               % Speed of light
lambda_wave = para.lambda_wave;   % Carrier wavelength

% Frequency grid over the selected subcarriers
f_n = para.fc + (0:para.N-1) * para.deltaf;
% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);
% Generate clutter channels and related parameters
[H_cold0, H_hot0, H_uav0, uavPara, scatterPara, coldPara, hotPara] = ...
    gen_clutter_channels(para, hotSourcePara, tarPara, cfg);
hotPara.sigma2_mu_hn = abs(hotPara.mu_hn).^2;
%% Construct the ideal spatial clutter kernel Vcc
theta_cc_all = [coldPara.thetas(:); uavPara.thetas(:)];
sigma2_cc_all = [coldPara.sigma2_beta_cn(:,1:N); abs(uavPara.beta_cn(:,1:N)).^2];
[theta_cc_all, idx_sort] = sort(theta_cc_all, 'ascend');
sigma2_cc_all = sigma2_cc_all(idx_sort,:);

Vcc = cell(N,1);
Call = numel(theta_cc_all);
for n = 0:(N-1)
    Vcc_n = zeros(Nr*Nr, Nt*Nt);
    for c = 1:Call
        a = a_n(theta_cc_all(c), n);
        b = b_n(theta_cc_all(c), n);
        vec_bbH = kron(conj(b), b);
        vec_aaH = kron(conj(a), a);
        Vcc_n = Vcc_n + sigma2_cc_all(c, n+1) * (vec_bbH * vec_aaH');
    end
    Vcc{n+1} = Vcc_n;
end
%% estimate the clutter covariance Rcc from clutter echoes
Nprobe = 400;        % Number of probing transmit covariance samples
Jclutter = 5e3;      % Number of Monte Carlo clutter realizations

% Store random probing beamformers and their corresponding transmit covariances
W_bank = cell(Nprobe, 1);
Rxx_mat = complex(zeros(Nt * Nt, Nprobe, N));
for ip = 1:Nprobe
    W_total = (randn(Nt, Ns, N) + 1j * randn(Nt, Ns, N)) / sqrt(2);
    W_total = W_total * sqrt(Pt / norm(W_total, 'fro')^2);
    for n = 1:N
        Rxx_mat(:, ip, n) = reshape(W_total(:, :, n) * W_total(:, :, n)', [], 1);
    end
    W_bank{ip} = W_total;
end

Rcc_acc = complex(zeros(Nr*Nr, Nprobe, N));
for jc = 1:Jclutter
    H_cold_j = draw_H_cold_realization(para, coldPara);
    Rcc_one = complex(zeros(Nr*Nr, Nprobe, N));
    parfor ip = 1:Nprobe
        W_total = W_bank{ip};
        bits_sens = randi([0,3], Ns, N, L);
        s_sens = qammod(bits_sens, 4, 'UnitAveragePower', true);
        Rcc_local = complex(zeros(Nr*Nr, N));
        for n = 1:N
            Yn = complex(zeros(Nr, L));
            for l = 1:L
                x = W_total(:,:,n) * squeeze(s_sens(:,n,l));
                Yn(:,l) = (H_cold_j(:,:,l,n) + H_uav0(:,:,l,n)) * x;
            end
            R_emp = (Yn * Yn') / L;
            R_emp = (R_emp + R_emp') / 2;
            Rcc_local(:,n) = R_emp(:);
        end
        Rcc_one(:,ip,:) = reshape(Rcc_local, Nr*Nr, 1, N);
    end
    Rcc_acc = Rcc_acc + Rcc_one;
end
Rcc_mat = Rcc_acc / Jclutter;
%%  Estimate spatial kernel Vcc from the clutter covariance
Vcc_hat = cell(N,1);
for n = 1:N
    Xn = Rxx_mat(:,:,n);
    Yn = Rcc_mat(:,:,n);
    Vcc_temp = Yn*pinv(Xn);
    Vcc_hat{n} = project_Vcc_to_CP(Vcc_temp, Nr, Nt, 1e-15);
end
%% Estimate the interference-plus-noise covariance R_eta
Jhot = 1e4;
Reta_cell = cell(Jhot,1);
parfor jh = 1:Jhot
    Reta_local = zeros(Nr, Nr, N);
    x_hs = sqrt(hotSourcePara.ILpower(1) / N / Nt_hs / 2) * ...
        (randn(Nt_hs, N, L) + 1j*randn(Nt_hs, N, L));
    y_hot = complex(zeros(Nr, N, L));
    for l = 1:L
        for n = 1:N
            y_hot(:,n,l) = H_hot0(:,:,l,n) * x_hs(:,n,l);
        end
    end
    noise = sqrt(para.sigma2/2) * (randn(Nr, N, L) + 1j*randn(Nr, N, L));
    y_eta = y_hot + noise;
    for n = 1:N
        Yn = reshape(y_eta(:,n,:), Nr, L);
        Rn = (Yn * Yn') / L;
        Rn = (Rn + Rn') / 2;
        Reta_local(:,:,n) = Rn;
    end
    Reta_cell{jh} = Reta_local;
end
R_eta = zeros(Nr, Nr, N);
for jh = 1:Jhot
    R_eta = R_eta + Reta_cell{jh};
end
R_eta = R_eta / Jhot;
%% Save the generated covariance matrices
save("data\covMat_BLP.mat", 'para', 'tarPara', 'uavPara', 'coldPara', ...
    'Vcc', 'Vcc_hat', 'R_eta', 'theta_cc_all', 'sigma2_cc_all', 'Jhot', 'Jclutter');