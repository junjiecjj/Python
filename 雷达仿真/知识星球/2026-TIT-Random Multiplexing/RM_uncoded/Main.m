% ----------------------------------------------------------------
% Uncoded case: random multiplexing (RM) v.s. OFDM/OTFS/AFDM
% Authors: Lei Liu, Shunqi Huang, Yuhao Chi, Yao Ge
% ----------------------------------------------------------------
% If you have any questions, please contact:
% lei_liu@zju.edu.cn
% ----------------------------------------------------------------
% If you use this code, please consider citing:
% [1] L. Liu, Y. Chi, S. Huang, Z. Zhang, "Random Multiplexing," in IEEE Trans. 
% Inf. Theory, Accepted in 2025.
% [2] L. Liu, Y. Chi, S. Huang, "Random Modulation: Achieving Asymptotic 
% Replica Optimality over Arbitrary Norm-Bounded and Spectrally Convergent 
% Channel Matrices," in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Michigan, USA, 2025.
% [3] Y. Chi, L. Liu, Y. Ge, X. Chen, Y. Li, and Z. Zhang, "Interleave Frequency 
% Division Multiplexing," IEEE Wireless Communications Letters, 
% vol. 13, no. 7, pp. 1963 - 1967, 2024.
% [4] L. Liu, S. Huang, and B. M. Kurkoski, "Memory AMP," 
% IEEE Transactions on Information Theory, vol. 68, no. 12, pp. 8015-8039, 2022.
% [5] S. Huang, L. Liu and B. M. Kurkoski, "Overflow-Avoiding Memory AMP," 
% in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Athens, Greece, 2024, pp. 3516-3521

%% Parameters
clc; clear;
rng('shuffle');
% Multipath channel parameters
P = 5;                      % Number of Path
delta_f = 1.5e4;            % Subcarrier spacing 
M = 32;                     % total delay span > channel's maximum delay
N = 32;                     % total Doppler span > channel's maximum Doppler shift
MN = M * N;                                       
vel = 150;                              % Velocity
dop = vel * (1e3/3600) * (4e9/3e8);     % Doppler frequency shift
index_D = 1;                            % 1 means Doppler shift
fs_N = 1;                               % >1 means oversampling
fs = fs_N * M * delta_f;                % Sampling rate
beta = 0.4;                             % Roll-off factor of raised-cosine filter
Ns = 2;                                 % Number of transmit antennas
Nr = 2;                                 % Number of received antennas
N_y = Nr * MN;          
N_x = Ns * MN; 
rho = 0.3;                              % MIMO correlation factor    
L = 3;                                  % damping length, used in MAMP
info = struct('type', "QPSK", 'mean', 0, 'var', 1);     % See 'Demodulator.m'
iter_O = 10;
iter_M = 30;
% Modulations (for OFDM and AFDM, we use the per-antenna modulation scheme)
otfs_info = struct('type', "OTFS", 'M', M, 'N', N, 'N_s', Ns);
ofdm_info = struct('type', 'OFDM_p', 'MN', MN, 'N_s', Ns);
Epsilon = N;                            % 0 (>0): integer (fractional) Doppler shift
Doppler_taps_max = round(dop*N/delta_f);
c1 = (2*(Doppler_taps_max+Epsilon)+1) / (2*MN);
c2 = 1e-5;                              % c2 should be much smaller than 1/(2*MN) 
afdm_info = struct('type', "AFDM_p", 'MN', MN, 'N_s', Ns, 'c1', c1, 'c2', c2);
none_info = struct('type', "None");     % No modulation

%% Simulations
SNR_dB = 4:2:24;
len = length(SNR_dB);
BER_ofdm = zeros(1, len);
BER_otfs = zeros(1, len);
BER_afdm = zeros(1, len);
BER_rm = zeros(1, len);
BER_none = zeros(1, len);
Var_se = zeros(1, len);
Num_sim = [100, 100, 500, 500, 5000*ones(1, len-4)];
poolobj = gcp('nocreate');     
if isempty(poolobj)
    poolsize = 0;
    CoreNum = 10;
    parpool(CoreNum);
end
for ii = 1 : len
    snr = SNR_dB(ii);
    v_n = 1 / (10^(0.1*snr));
    [E_1, E_2, E_3, E_4, E_5, Var] = deal(0);
    fprintf('---------------SNR: %ddB--------------- \n', snr);
    parfor jj = 1 : Num_sim(ii)
        disp(jj)
        % QPSK signal
        d = binornd(1, 0.5, 2*N_x, 1);
        s = Bits_to_QPSK(d);
        % Time-domain multipath channel 
        H = Get_channel_sparse(M, N, Nr, Ns, rho, fs, fs_N, P, index_D, dop, beta);
        H(abs(H)<1e-8) = 0;
        [~, dia, V] = svd(full(H));         % necessary for OAMP
        dia = diag(dia);
        temp = sum(dia.^2) / N_x;
        H = H / sqrt(temp);                 % channel normalization 
        dia = dia / sqrt(temp);             % channel normalization 
        % Gaussian noise
        n_re = normrnd(0, sqrt(v_n/2), [N_y, 1]); 
        n_im = normrnd(0, sqrt(v_n/2), [N_y, 1]);
        n = n_re + n_im * 1i;
        % OFDM, OTFS, AFDM
        x_ofdm = Modulations(s, ofdm_info, 0);
        y_ofdm = H * x_ofdm + n;
        x_otfs = Modulations(s, otfs_info, 0);
        y_otfs = H * x_otfs + n;
        x_afdm = Modulations(s, afdm_info, 0);
        y_afdm = H * x_afdm + n;
        x_none = Modulations(s, none_info, 0);
        y_none = H * x_none + n;
        % RM: y = H * Xi * s + n, Xi = Pi * F 
        % Pi is a random permutation, F is a fast transform
        % Notice the RM type!
        index = randperm(N_x);
        rm_info = struct('type', "RM", 'rm_type', "fwht", 'N_x', N_x, 'index', index);
        x_rm = Modulations(s, rm_info, 0);
        y_rm = H * x_rm + n;
        % CD-OAMP detector
        % No demodulation is performed at the receiver before CD-OAMP detector.
        % Since demodulation never changes the performance of CD-OAMP.
        [~, ~, s_ofdm] = CD_OAMP(H, V, s, y_ofdm, dia, v_n, iter_O, info, ofdm_info);
        [~, ~, s_otfs] = CD_OAMP(H, V, s, y_otfs, dia, v_n, iter_O, info, otfs_info);
        [~, ~, s_afdm] = CD_OAMP(H, V, s, y_afdm, dia, v_n, iter_O, info, afdm_info);
        [~, ~, s_none] = CD_OAMP(H, V, s, y_none, dia, v_n, iter_O, info, none_info);
        % CD-MAMP detector
        [~, ~, s_rm] = CD_MAMP_e(H, s, y_rm, v_n, L, iter_M, info, rm_info);
        % State evolution of OAMP
        [~, V_le, ~] = OAMP_SE_qpsk(dia, v_n, iter_O, N_x);
        Var = Var + V_le(end);
        % Hard decision (bits)
        d_ofdm = QPSK_to_bits(s_ofdm);
        E_1 = E_1 + sum(d_ofdm~=d);
        d_otfs = QPSK_to_bits(s_otfs);
        E_2 = E_2 + sum(d_otfs~=d);
        d_afdm = QPSK_to_bits(s_afdm);
        E_3 = E_3 + sum(d_afdm~=d);
        d_rm = QPSK_to_bits(s_rm);
        E_4 = E_4 + sum(d_rm~=d);
        d_none = QPSK_to_bits(s_none);
        E_5 = E_5 + sum(d_none~=d);
    end
    BER_ofdm(ii) = E_1 / Num_sim(ii) / (2*N_x);
    BER_otfs(ii) = E_2 / Num_sim(ii) / (2*N_x);
    BER_afdm(ii) = E_3 / Num_sim(ii) / (2*N_x);
    BER_rm(ii) = E_4 / Num_sim(ii) / (2*N_x);
    BER_none(ii) = E_5 / Num_sim(ii) / (2*N_x);
    Var_se(ii) = Var / Num_sim(ii);
    fprintf('BER: \n');
    fprintf('OFDM + OAMP: %.6f \n', BER_ofdm(ii))
    fprintf('OTFS + OAMP: %.6f \n', BER_otfs(ii))
    fprintf('AFDM + OAMP: %.6f \n', BER_afdm(ii))
    fprintf('RM + MAMP %.6f \n', BER_rm(ii))
    fprintf('------------------------------ \n')
end

%% SE variance to BER (QPSK)
% r = x + w, w ~ CN(0, v), find its MAP BER 
% For high SNR(dB), use the union bound (MAP_QAM function)
BER_se = zeros(1, len);
K = 4;
for k = 1 : K
    v = Var_se(k);
    N = 1e6;
    d = binornd(1, 0.5, 2*N, 1);
    s = Bits_to_QPSK(d);
    n_re = normrnd(0, sqrt(v/2), [N, 1]); 
    n_im = normrnd(0, sqrt(v/2), [N, 1]);
    n = n_re + n_im * 1i;
    r = s + n;
    [s_se, ~] = Denoiser(r, v, info);
    d_se = QPSK_to_bits(s_se);
    BER_se(k) = sum(d_se~=d) / (2*N);
end
for k = K+1 : len
    rho_k = 10^(SNR_dB(k)/10);
    BER_se(k) = MAP_QAM(4, rho_k);    % QPSK: 4, 16QAM: 16
end

%% plot figures
semilogy(SNR_dB, BER_rm, 'r-', 'LineWidth', 1.5);
hold on;
semilogy(SNR_dB, BER_ofdm, '-', 'LineWidth', 1.5);
semilogy(SNR_dB, BER_otfs, '-', 'LineWidth', 1.5);
semilogy(SNR_dB, BER_afdm, '-', 'LineWidth', 1.5);
semilogy(SNR_dB, BER_none, '-', 'LineWidth', 1.5);
semilogy(SNR_dB, BER_se, 'o', 'LineWidth', 1.5);
ylim([8e-6, 0.2])
legend('RM + MAMP', 'OFDM + OAMP', 'OTFS + OAMP', 'AFDM + OAMP', 'None + OAMP', 'SE');
xlabel('SNR (dB)', 'FontSize', 11);

ylabel('BER', 'FontSize', 11);

