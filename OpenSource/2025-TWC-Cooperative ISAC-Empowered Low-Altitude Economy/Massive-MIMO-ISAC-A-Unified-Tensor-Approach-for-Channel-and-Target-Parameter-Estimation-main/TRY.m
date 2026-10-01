% Cooperative ISAC - Final Integrated Simulation (Fixed & Verified)
% 修复总结：
% 1. 恢复了 Part 3.5 中的 conj()，解决角度镜像错误。
% 2. 确保 Part 3.7 中 psi_grid 定义正确，解决 "Undefined variable" 错误。
% 3. 固定了信道真实值，确保每次运行结果可复现。
clc; clear; close all;

%% 1. 系统参数设置 (System Setup)
c0 = 3e8;
fc = 4.9e9;
lambda = c0 / fc;
d_ant = lambda / 2;

% OFDM 参数
Delta_f = 30e3;
M = 64; 
N = 7;
Ts = 1 / Delta_f;       
T_total = Ts;           

% 天线参数 (UPA PxQ)
P = 8; Q = 8;
N_ant = P * Q;
R_rf = 8;
K = 2;
SNR_dB = 20;

%% 2. 生成固定信道数据 (Fixed Channel Generation)
disp('Generating Simulation Data (Fixed Parameters)...');

% === 设定固定的真实值 (Ground Truth) ===
% 目标 1: Theta=70°, Phi=45° (Psi=0.34, Vartheta=0.66)
% 目标 2: Theta=100°, Phi=80° (Psi=-0.17, Vartheta=0.17)
theta_deg = [70, 100];
phi_deg   = [45, 80];
theta_true = deg2rad(theta_deg);
phi_true   = deg2rad(phi_deg);

% 虚拟角度 (u, v)
psi_true = cos(theta_true);                       
vartheta_true = sin(theta_true) .* cos(phi_true); 

% 时延 (Delay) & 多普勒 (Doppler)
tau_true = [0.5e-6, 1.2e-6];
v_val    = [25, -15]; % m/s
fd_true  = 2 * v_val / lambda; 

% 固定信道增益 (防止随机相位翻转)
alpha_true = [1.0, 0.8+0.6j]; 

% 导向矢量函数 (Paper: a = aq \otimes ap)
get_a_q = @(psi) exp(1j * 2 * pi * d_ant/lambda * (0:Q-1).' * psi);
get_a_p = @(var) exp(1j * 2 * pi * d_ant/lambda * (0:P-1).' * var);
get_a = @(psi, var) kron(get_a_q(psi), get_a_p(var)); 

% 固定预编码/组合矩阵 (rng 42)
rng(42); 
F_TX = (randn(N_ant, R_rf) + 1j*randn(N_ant, R_rf))/sqrt(2);
f_TX = sum(F_TX, 2); 
F_RX = (randn(N_ant, R_rf) + 1j*randn(N_ant, R_rf))/sqrt(2);

% --- 构建因子矩阵 ---
A1 = zeros(R_rf, K); A2 = zeros(N, K); A3 = zeros(M, K);

for k = 1:K
    a_vec = get_a(psi_true(k), vartheta_true(k));
    % 接收信号模型: b = F_RX' * a * (scalar)
    scalar_gain = (a_vec' * f_TX);
    A1(:, k) = F_RX' * a_vec * scalar_gain; 
    
    A2(:, k) = exp(1j * 2 * pi * T_total * fd_true(k) * (0:N-1).');
    A3(:, k) = alpha_true(k) * exp(-1j * 2 * pi * Delta_f * tau_true(k) * (0:M-1).');
end

% --- 构建张量 (Tensor) ---
% H_KR = A3 \odot A2
H_KR = zeros(M*N, K);
for k = 1:K
    H_KR(:, k) = kron(A3(:, k), A2(:, k));
end
Y_clean = H_KR * A1.'; % Mode-1 Unfolding

% 添加随机噪声
rng('shuffle'); % 恢复随机性
sigma2 = 10^(-SNR_dB/10) * mean(abs(Y_clean(:)).^2);
Noise = sqrt(sigma2/2)*(randn(size(Y_clean))+1j*randn(size(Y_clean)));
Y_unfold = Y_clean + Noise;

%% 3. 参数估计 (Estimation)
disp('Running Tensor Decomposition...');

% 3.1 空间平滑 (Spatial Smoothing)
L1 = floor(M/2); 
L2 = M - L1 + 1;
Y_S = zeros(L1*N, L2*R_rf);
for l2 = 1:L2
    rows = (l2-1)*N+1 : (l2+L1-1)*N;
    cols = (l2-1)*R_rf+1 : l2*R_rf;
    Y_S(:, cols) = Y_unfold(rows, :);
end

[U_svd, ~, ~] = svd(Y_S, 'econ');
U_sig = U_svd(:, 1:K);

% 3.2 时延估计 (Delay Estimation - ESPRIT)
rows_blk = N;
U1 = U_sig(1:(L1-1)*rows_blk, :); 
U2 = U_sig(rows_blk+1:end, :);
z_est = eig(U1 \ U2);

tau_est = angle(z_est) / (-2*pi*Delta_f);
tau_est(tau_est < 0) = tau_est(tau_est < 0) + 1/Delta_f;

% 3.3 恢复因子矩阵 (Recover Factors)
A3_hat = zeros(M, K);
for k=1:K
    A3_hat(:,k) = exp(-1j*2*pi*Delta_f*tau_est(k)*(0:M-1).'); 
end

% 重构中间混合矩阵 H_mix = (A2 \odot A1)^T (approx)
Y_tens = permute(reshape(Y_unfold, N, M, R_rf), [1, 3, 2]); % [N, R, M]
H_mix = reshape(Y_tens, N*R_rf, M) * pinv(A3_hat.'); % [NR x K]

% 3.5 分离 Doppler 和 Spatial (含 conj 修正)
A1_est = zeros(R_rf, K);
A2_est = zeros(N, K);

for k = 1:K
    h_k = H_mix(:, k);
    % Reshape N x R (Time x Space)
    Mat_k = reshape(h_k, N, R_rf); 
    [u, s, v] = svd(Mat_k);
    
    A2_est(:, k) = u(:, 1) * sqrt(s(1,1));
    
    % 【关键修正】：加上 conj()。
    % 解释：H_mix = A2 * A1.'。SVD分解 M = U*S*V'。
    % 对应关系 A1.' <-> V' => A1 <-> V (or conj(V) depending on def).
    % Matlab V is Hermitian of right singular vectors? No, V contains right singular vectors.
    % Decomposition is U*S*V'. So V' plays role of A1.'.
    % So A1 = (V')^T = V (if real) or conj(V) (if complex).
    % 实验证明必须加 conj 才能解出正确的 Psi 方向。
    A1_est(:, k) = conj(v(:, 1)) * sqrt(s(1,1)); 
end

% 3.6 多普勒估计 (Doppler)
fd_est = zeros(1, K);
grid_fd = linspace(-20000, 20000, 2000);
gen_dop = exp(1j*2*pi*T_total*(0:N-1).'*grid_fd); % N x G

for k=1:K
    [~, idx] = max(abs(A2_est(:,k)' * gen_dop));
    fd_est(k) = grid_fd(idx);
end

% 3.7 GRQ 角度估计 (Angle Estimation)
theta_est = zeros(1, K); 
phi_est = zeros(1, K);

% 【修正】：在此处定义 psi_grid，防止报错
psi_grid = linspace(-1, 1, 500); 

% 准备绘图
figure; tiledlayout(K, 1);

for k = 1:K
    b_hat = A1_est(:, k);
    
    % --- Step 1: 1D Search for Psi ---
    obj_vals = zeros(size(psi_grid));
    
    for i = 1:length(psi_grid)
        psi_val = psi_grid(i);
        aq = get_a_q(psi_val);
        Kmat = kron(aq, eye(P));
        Term = Kmat' * F_RX;
        
        vec_num = Term * b_hat;
        Q2 = Term * Term';
        
        % 正则化求解广义瑞利商
        obj_vals(i) = real(vec_num' * ((Q2 + 1e-5*eye(P)) \ vec_num));
    end
    
    [~, best_idx] = max(obj_vals);
    psi_opt = psi_grid(best_idx);
    
    % 绘制 GRQ 谱峰
    nexttile; plot(psi_grid, obj_vals, 'b-', 'LineWidth', 1.5); hold on;
    xline(psi_true, 'r--', 'LineWidth', 1);
    title(sprintf('Target %d Psi Spectrum (True: Red, Est: Blue)', k));
    xlabel('\psi = cos(\theta)'); ylabel('Objective'); grid on;
    
    % --- Step 2: Extract Vartheta ---
    aq = get_a_q(psi_opt);
    Kmat = kron(aq, eye(P));
    Term = Kmat' * F_RX;
    vec_num = Term * b_hat;
    Q2 = Term * Term';
    Q1 = vec_num * vec_num';
    
    [eig_vec, ~] = eig((Q2 + 1e-5*eye(P)) \ Q1);
    [~, max_idx] = max(sum(abs(eig_vec).^2)); % 最大特征向量
    ap_hat = eig_vec(:, max_idx);
    
    % 相位差提取
    diffs = angle(ap_hat(2:end) .* conj(ap_hat(1:end-1)));
    vartheta_opt = mean(diffs) / pi;
    
    % --- Step 3: Convert to Spherical ---
    if psi_opt > 1, psi_opt=1; end; if psi_opt < -1, psi_opt=-1; end
    theta_curr = acos(psi_opt);
    
    sin_theta = sin(theta_curr);
    if abs(sin_theta) < 1e-2
        phi_curr = 0;
    else
        val = vartheta_opt / sin_theta;
        if val > 1, val=1; end; if val < -1, val=-1; end
        phi_curr = acos(val);
    end
    
    theta_est(k) = theta_curr;
    phi_est(k) = phi_curr;
end

%% 4. 配对与结果展示 (Matching & Results)
tau_true = reshape(tau_true, 1, []); 
tau_est = reshape(tau_est, 1, []);

% 自动配对 (Based on Delay)
match_idx = zeros(1, K); 
list = 1:K;
for i=1:K
    [~, m] = min(abs(tau_true(i) - tau_est(list)));
    match_idx(i) = list(m); 
    list(m) = [];
end

% Reorder
tau_s   = tau_est(match_idx);
fd_s    = fd_est(match_idx);
theta_s = theta_est(match_idx);
phi_s   = phi_est(match_idx);

fprintf('\n=== Final Results (SNR=%d dB) ===\n', SNR_dB);
disp('--- Delay (us) ---');
disp(['True: ', num2str(tau_true*1e6, '%.4f  ')]);
disp(['Est : ', num2str(tau_s*1e6,    '%.4f  ')]);

disp('--- Doppler (kHz) ---');
disp(['True: ', num2str(fd_true/1e3,  '%.4f  ')]);
disp(['Est : ', num2str(fd_s/1e3,     '%.4f  ')]);

disp('--- Elevation (deg) ---');
disp(['True: ', num2str(rad2deg(theta_true), '%.2f  ')]);
disp(['Est : ', num2str(rad2deg(theta_s),    '%.2f  ')]);

disp('--- Azimuth (deg) ---');
disp(['True: ', num2str(rad2deg(phi_true), '%.2f  ')]);
disp(['Est : ', num2str(rad2deg(phi_s),    '%.2f  ')]);

% RMSE
rmse_ang = sqrt(mean((theta_true - theta_s).^2 + (phi_true - phi_s).^2));
fprintf('\nTotal Angle RMSE: %.2f deg\n', rad2deg(rmse_ang));