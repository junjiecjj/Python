%% 通用代码：适用于任意 M 和任意 L（图8 为 L=2）

clc;
clear all;
close all;
rng(42); 

%% 用户参数
M = 2;                          % 阵元数
L = 2;                          % 目标数
d_lambda = 0.5;                 % 半波长间距
SNR_dB = 0;                     % 信噪比 (dB)
SNR_lin = 10^(SNR_dB/10);
N = 1;                          % 快拍数
alphas = ones(L, 1);            % 所有目标复振幅设为1
sigma_w2 = N * abs(alphas(1))^2 / SNR_lin;   % 噪声方差

theta_deg = [0, 5];             % 两个目标角度（第一个固定0°，第二个可变）
theta2_list = [5, 10, 15];      % 用于图8的第二个目标角度列表
beta_vals = linspace(0, 0.9999, 100);

% 对称阵列（质心在原点），用于满足论文假设
n = 0:M-1;         % 阵元位置
a = @(th) exp(-1j * 2*pi*d_lambda * n' * sind(th));
da = @(th) -1j * 2*pi*d_lambda * cosd(th) * n' .* a(th);
A_mat = @(th) a(th) * a(th).';
dA_dtheta = @(th) da(th) * a(th).' + a(th) * da(th).';

% 等效导向矢量 d_β(θ) = sqrt(N) * vec( A(θ) * U Λ^{1/2} )   -> 4×1 列向量
d_beta_vec = @(theta_deg, beta) reshape( sqrt(N) * (A_mat(theta_deg) * get_U_sqrtL(beta, M)), [], 1);
% d_β(θ) 对角度（弧度）的导数 -> 4×1 列向量
d_dbeta_dtheta_vec = @(theta_deg, beta) reshape( sqrt(N) * (dA_dtheta(theta_deg) * get_U_sqrtL(beta, M)), [], 1);


% 预分配结果（存储 θ1 的标准差，单位度）
CRB_results = zeros(length(theta2_list), length(beta_vals));

for k = 1:length(theta2_list)
    theta2 = theta2_list(k);
    theta_deg(2) = theta2;      % 更新第二个目标角度
    for b = 1:length(beta_vals)
        beta = beta_vals(b);
        % 构建每个目标的等效导向矢量及其导数（列向量）
        d_vecs = cell(L, 1);
        d_deriv_vecs = cell(L, 1);
        for l = 1:L
            th = theta_deg(l);
            d_vecs{l} = d_beta_vec(th, beta);
            d_deriv_vecs{l} = d_dbeta_dtheta_vec(th, beta);
        end
        % 梯度矩阵 G：每列对应一个实参数
        % 参数顺序: [θ1, θ2, ..., θL, Re(α1), Im(α1), ..., Re(αL), Im(αL)]
        n_params = L + 2*L;      % 角度 L 个，复振幅实虚部 2L 个
        obs_dim = M^2;           % 等效观测维度
        G = zeros(obs_dim, n_params);
        % 角度部分
        for l = 1:L
            G(:, l) = alphas(l) * d_deriv_vecs{l};
        end
        % 复振幅部分
        for l = 1:L
            G(:, L + 2*(l-1) + 1) = d_vecs{l};        % Re(α_l)
            G(:, L + 2*(l-1) + 2) = 1j * d_vecs{l};    % Im(α_l)
        end
        
        % Fisher 信息矩
        J = (2/sigma_w2) * real(G' * G);
        
        % 求逆得 CRB 矩阵，提取第一个角度 θ1 的标准差（度）
        if rcond(J) > 1e-12
            J_inv = inv(J);
            CRB_theta1_rad2 = J_inv(1,1);
            % CRB_deg = sqrt(CRB_theta1_rad2) * 180/pi;
            CRB_deg = CRB_theta1_rad2;
        else
            CRB_deg = NaN;
        end
        CRB_results(k, b) = CRB_deg;
    end
end

% 绘图
figure(1);
colors = {'b','r','g'};
for k = 1:length(theta2_list)
    semilogy(beta_vals, CRB_results(k,:), 'Color', colors{k}, 'LineWidth', 2, ...
         'DisplayName', sprintf('\\theta_2 = %d°', theta2_list(k)));
    hold on;
end
xlabel('\beta'); ylabel('CRB (deg)');
title(sprintf('Figure 8: M=%d, L=%d, SNR=0dB', M, L));
grid on; legend('Location','best');

% 通用函数：计算 U*sqrt(Lambda)
function U_sqrtL = get_U_sqrtL(beta, M)
    Rs = (1-beta)*eye(M) + beta*ones(M);
    [U, Lambda] = eig(Rs);
    lambda = diag(Lambda);
    lambda_sqrt = sqrt(max(lambda, 0));
    U_sqrtL = U * diag(lambda_sqrt);
end