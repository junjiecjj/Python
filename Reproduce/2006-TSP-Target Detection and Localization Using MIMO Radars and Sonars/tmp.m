%% 复现 Bekkerman & Tabrikian (2006) Figure 9
% M=10 阵元，双目标：θ1=0° 固定，θ2 从 0° 到 2° 变化，SNR=0dB
% 相干信号: 发射波束固定指向 0° (R_s = a(0°)a^H(0°))
% 正交信号: 全向发射 (R_s = I)，TOT 补偿使有效 SNR 提高 M 倍
% 绘制 CRB 和 ML 估计的 RMSE (200 次蒙特卡洛，二维网格搜索)

clear; clc; close all;

%% 参数设置
M = 10;                     % 阵元数
d_lambda = 0.5;             % 半波长间距
SNR_dB = 0;                 % 基础信噪比 (0 dB)
SNR_lin = 10^(SNR_dB/10);   % 线性值
N = 1;                      % 快拍数（可任意，CRB 与 N 成反比，这里取 1）
alpha1 = 1;                 % 目标1复振幅
alpha2 = 1;                 % 目标2复振幅

% 相干信号：发射波束指向 0°，R_s = a_beam * a_beam^H
beam_theta = 0;             % 发射波束方向（度）
% 正交信号：R_s = I，TOT 补偿使等效噪声方差降低 M 倍
sigma_w2_coherent = 1 / SNR_lin;        % 相干信号噪声方差
sigma_w2_orth = sigma_w2_coherent / M;  % 正交信号噪声方差 (TOT 补偿)

% 对称阵列（质心为原点）
n_idx = -(M-1)/2 : (M-1)/2;             % -4.5:4.5
a = @(theta_deg) exp(-1j * pi * d_lambda * n_idx' * sind(theta_deg));
% 导向矢量对角度（度）的导数，用于 CRB
da_dtheta_deg = @(theta_deg) -1j * pi * d_lambda * cosd(theta_deg) * n_idx' .* a(theta_deg);

% 相干信号的发射导向矢量
a_beam = a(beam_theta);
R_s_coherent = a_beam * a_beam';
R_sT_coherent = R_s_coherent.';

% 正交信号的 R_s = I
R_sT_orth = eye(M);

%% 等效导向矢量 d_β(θ) 及其导数 (长度 M^2 = 100)
% 对于相干信号，计算 U Λ^{1/2}
[U_coherent, Lambda_coherent] = eig(R_s_coherent);
Lambda_sqrt_coherent = sqrt(Lambda_coherent);
U_sqrtL_coherent = U_coherent * Lambda_sqrt_coherent;   % 秩1，第二列为0
% 对于正交信号，UΛ^{1/2} = I
U_sqrtL_orth = eye(M);

% 函数：计算 d_β(θ) = sqrt(N) * vec( A(θ) * UΛ^{1/2} )
d_beta = @(theta_deg, U_sqrtL) reshape( sqrt(N) * (a(theta_deg) * a(theta_deg).' * U_sqrtL), [], 1);
% 导数 d_β'(θ) = sqrt(N) * vec( (a'(θ)a^T(θ) + a(θ)a'^T(θ)) * UΛ^{1/2} )
d_beta_deriv = @(theta_deg, U_sqrtL) reshape( sqrt(N) * ( (da_dtheta_deg(theta_deg) * a(theta_deg).' + ...
                                                         a(theta_deg) * da_dtheta_deg(theta_deg).') * U_sqrtL ), [], 1);

%% 计算 CRB (解析)
theta2_deg_vec = linspace(0.1, 2, 20);   % 0~2度，避开0度奇异
CRB_coherent = zeros(size(theta2_deg_vec));
CRB_orth = zeros(size(theta2_deg_vec));

for idx = 1:length(theta2_deg_vec)
    theta2 = theta2_deg_vec(idx);
    theta_deg = [0; theta2];
    
    % --- 相干信号 CRB ---
    d1 = d_beta(theta_deg(1), U_sqrtL_coherent);
    d2 = d_beta(theta_deg(2), U_sqrtL_coherent);
    d1p = d_beta_deriv(theta_deg(1), U_sqrtL_coherent);
    d2p = d_beta_deriv(theta_deg(2), U_sqrtL_coherent);
    
    grad = zeros(length(d1), 6);
    grad(:,1) = alpha1 * d1p;
    grad(:,2) = alpha2 * d2p;
    grad(:,3) = d1;
    grad(:,4) = 1j * d1;
    grad(:,5) = d2;
    grad(:,6) = 1j * d2;
    J = (2/sigma_w2_coherent) * real(grad' * grad);
    if rcond(J) > 1e-12
        CRB_var = inv(J);
        CRB_coherent(idx) = sqrt(CRB_var(1,1)) * (180/pi);
    else
        CRB_coherent(idx) = NaN;
    end
    
    % --- 正交信号 CRB ---
    d1 = d_beta(theta_deg(1), U_sqrtL_orth);
    d2 = d_beta(theta_deg(2), U_sqrtL_orth);
    d1p = d_beta_deriv(theta_deg(1), U_sqrtL_orth);
    d2p = d_beta_deriv(theta_deg(2), U_sqrtL_orth);
    
    grad = zeros(length(d1), 6);
    grad(:,1) = alpha1 * d1p;
    grad(:,2) = alpha2 * d2p;
    grad(:,3) = d1;
    grad(:,4) = 1j * d1;
    grad(:,5) = d2;
    grad(:,6) = 1j * d2;
    J = (2/sigma_w2_orth) * real(grad' * grad);
    if rcond(J) > 1e-12
        CRB_var = inv(J);
        CRB_orth(idx) = sqrt(CRB_var(1,1)) * (180/pi);
    else
        CRB_orth(idx) = NaN;
    end
end

%% 蒙特卡洛 ML 仿真 (RMSE)
MC_trials = 200;                     % 每个 θ2 点的仿真次数
theta2_ml_vec = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0];  % 选取几个点，避免过密计算
RMSE_coherent = zeros(size(theta2_ml_vec));
RMSE_orth = zeros(size(theta2_ml_vec));

% 二维网格搜索参数（角度范围覆盖真实值附近）
grid_res = 0.02;                    % 度
theta1_grid = -1:grid_res:1;        % 目标1可能范围（真实0°附近）
theta2_grid = @(true_theta2) true_theta2 + (-1:grid_res:1);  % 目标2范围随真实值变化

fprintf('开始蒙特卡洛 ML 仿真 (共 %d 个 θ2 点，每个点 %d 次)...\n', length(theta2_ml_vec), MC_trials);

for pt = 1:length(theta2_ml_vec)
    theta2_true = theta2_ml_vec(pt);
    % 当前目标2的搜索网格
    t2_grid = theta2_true + (-0.8:grid_res:0.8);   % 范围 ±0.8°
    t1_grid = -0.8:grid_res:0.8;
    
    % 存储估计值
    theta1_est_coherent = zeros(MC_trials, 1);
    theta1_est_orth = zeros(MC_trials, 1);
    
    for mc = 1:MC_trials
        % --- 相干信号数据生成 ---
        A1 = a(0) * a(0).';
        A2 = a(theta2_true) * a(theta2_true).';
        % 发射信号 s[n] = a_beam (所有快拍相同)
        s_coherent = a_beam;    % M x 1
        % 无噪声接收
        signal_coherent = alpha1 * A1 * s_coherent + alpha2 * A2 * s_coherent;
        % 加噪声
        w_coherent = sqrt(sigma_w2_coherent/2) * (randn(M,1) + 1j*randn(M,1));
        y_coherent = signal_coherent + w_coherent;
        % 充分统计量 E = y * s^H / sqrt(N) = y * s_coherent^H (N=1)
        E_coherent = y_coherent * s_coherent';
        
        % 相干信号 ML 估计：二维网格搜索最大化 L(θ1,θ2) = eta^H P_D eta
        % 但更简单：计算每个网格点的统计量 L_grid = |a^H E a*|^2 / (M a^H R_s^T a)
        % 注意对于双目标，正确的 ML 是最大化投影矩阵的迹，但这里我们直接利用：
        % 对于给定的 (θ1,θ2)，构造 D = [d_beta(θ1), d_beta(θ2)]，然后 L = eta^H P_D eta
        % 其中 eta = vec(E * UΛ^{-1/2})? 但为了简化，我们采用单目标形式并分别估计两个角度
        % 实际上对于双目标，需联合估计。由于目标1角度固定为0附近，我们在此网格搜索二维似然。
        % 构造等效充分统计量 eta = vec(E * UΛ^{-1/2}) 对相干信号。
        % 计算 UΛ^{-1/2}: 伪逆，对于秩1矩阵
        [Uc, Lc] = eig(R_s_coherent);
        Lc_inv_sqrt = diag(1./sqrt(diag(Lc)));  % 奇异值倒数
        U_sqrtL_inv = Uc * Lc_inv_sqrt;
        eta_coherent = vec(E_coherent * U_sqrtL_inv);   % 100x1
        
        % 网格搜索
        best_L = -inf;
        best_theta1 = NaN;
        for i = 1:length(t1_grid)
            th1 = t1_grid(i);
            d1 = d_beta(th1, U_sqrtL_coherent);
            for j = 1:length(t2_grid)
                th2 = t2_grid(j);
                d2 = d_beta(th2, U_sqrtL_coherent);
                D = [d1, d2];
                P_D = D / (D'*D) * D';
                L_val = real(eta_coherent' * P_D * eta_coherent);
                if L_val > best_L
                    best_L = L_val;
                    best_theta1 = th1;
                end
            end
        end
        theta1_est_coherent(mc) = best_theta1;
        
        % --- 正交信号数据生成 ---
        % 发射信号矩阵 S (M x N), N=1 时，只需一个快拍，发射向量 s_orth 应满足 R_s = I
        % 即 E[s s^H] = I，最简单的 s 是任意单位向量乘以 sqrt(M) 以保证功率 M
        % 由于 N=1，无法实现正交集，但理论上仍然可以使 s 的协方差为 I: 例如取 s = sqrt(M)*randn(M,1)+1j...归一化。
        % 为保证 R_s = I，我们生成一个随机向量，其样本协方差期望为 I。这里取 s_orth = sqrt(M) * (randn(M,1)+1j*randn(M,1))/sqrt(2)
        s_orth = sqrt(M) * (randn(M,1) + 1j*randn(M,1)) / sqrt(2);
        signal_orth = alpha1 * A1 * s_orth + alpha2 * A2 * s_orth;
        w_orth = sqrt(sigma_w2_orth/2) * (randn(M,1) + 1j*randn(M,1));
        y_orth = signal_orth + w_orth;
        E_orth = y_orth * s_orth';
        % 对于正交信号，R_s = I, UΛ^{-1/2} = I，所以 eta = vec(E_orth)
        eta_orth = reshape(E_orth, [], 1);
        
        best_L = -inf;
        best_theta1 = NaN;
        for i = 1:length(t1_grid)
            th1 = t1_grid(i);
            d1 = d_beta(th1, U_sqrtL_orth);
            for j = 1:length(t2_grid)
                th2 = t2_grid(j);
                d2 = d_beta(th2, U_sqrtL_orth);
                D = [d1, d2];
                P_D = D / (D'*D) * D';
                L_val = real(eta_orth' * P_D * eta_orth);
                if L_val > best_L
                    best_L = L_val;
                    best_theta1 = th1;
                end
            end
        end
        theta1_est_orth(mc) = best_theta1;
    end
    
    % 计算 RMSE (估计误差，真实值为0)
    rmse_c = sqrt(mean((theta1_est_coherent - 0).^2));
    rmse_o = sqrt(mean((theta1_est_orth - 0).^2));
    RMSE_coherent(pt) = rmse_c;
    RMSE_orth(pt) = rmse_o;
    fprintf('θ2 = %.1f° : Coherent RMSE = %.3f°, Orthogonal RMSE = %.3f°\n', theta2_true, rmse_c, rmse_o);
end

%% 绘图
figure('Position', [100, 100, 600, 450]);
% 绘制 CRB 曲线（连续）
plot(theta2_deg_vec, CRB_coherent, 'b-', 'LineWidth', 2, 'DisplayName', 'Coherent CRB');
hold on;
plot(theta2_deg_vec, CRB_orth, 'r--', 'LineWidth', 2, 'DisplayName', 'Orthogonal CRB');
% 绘制 ML 估计 RMSE 点
plot(theta2_ml_vec, RMSE_coherent, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'Coherent ML');
plot(theta2_ml_vec, RMSE_orth, 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Orthogonal ML');

xlabel('Separation angle \theta_2 (deg)');
ylabel('RMSE / CRB (deg)');
title('Figure 9: DOA estimation of first target (M=10, L=2, SNR=0dB)');
grid on; grid minor;
legend('Location', 'best');
xlim([0, 2]);
ylim([0, 0.25]);
hold off;

%% 辅助函数: vec 操作
function v = vec(X)
    v = X(:);
end