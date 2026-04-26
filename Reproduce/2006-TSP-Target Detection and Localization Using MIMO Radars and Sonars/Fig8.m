


clc;
clear all;
close all;
% addpath('./functions');

rng(42); 


%% 复现 Bekkerman & Tabrikian (2006) Figure 8
% MIMO雷达中两个目标的角度估计CRB，M=2阵元，SNR=0dB
% 目标1固定于0°，目标2分别位于5°,10°,15°，相关系数β从0变化到1

%% 系统参数
M = 2;                      % 阵元数
d_lambda = 0.5;             % 阵元间距（波长归一化）
SNR_dB = 0;                 % 信噪比 (dB)
SNR_lin = 10^(SNR_dB/10);   % 线性值: SNR = N|α|^2/σ_w^2
N = 1;                      % 快拍数（可吸收进SNR，取1简化）
alpha1 = 1;                 % 目标1复振幅幅度
alpha2 = 1;                 % 目标2复振幅幅度
sigma_w2 = N * abs(alpha1)^2 / SNR_lin;  % 噪声方差 (SNR定义见论文(44))

% 角度（度）变换为弧度
theta1_deg = 0;             % 目标1角度
theta2_deg_list = [5, 10, 15];  % 目标2角度列表
beta_vals = linspace(0, 0.99, 200);  % 相关系数β (0~0.99，避开β=1奇异)

% 预分配结果
CRB_results = zeros(length(theta2_deg_list), length(beta_vals));

%% 辅助函数
% 导向矢量 a(θ) (均匀线阵，半波长间距)
a_func = @(theta_deg) [1; exp(-1j * pi * sind(theta_deg))];

% 导向矢量对角度（弧度）的导数 da/dθ
da_dtheta = @(theta_deg) [0; -1j * pi * cosd(theta_deg) * exp(-1j * pi * sind(theta_deg))];

% A(θ) = a(θ) * a(θ)^T
A_mat = @(theta_deg) a_func(theta_deg) * a_func(theta_deg).';

% A'(θ) = a' a^T + a a'^T
dA_dtheta = @(theta_deg) da_dtheta(theta_deg) * a_func(theta_deg).' + ...
                         a_func(theta_deg) * da_dtheta(theta_deg).';

% 计算 U * Λ^{1/2} 矩阵 (2x2)
% R_s = [1, beta; beta, 1] 的特征分解
U_sqrtLambda = @(beta) (1/sqrt(2)) * [sqrt(1+beta), sqrt(1-beta); 
                                       sqrt(1+beta), -sqrt(1-beta)];

% 等效导向矢量 d_β(θ) = sqrt(N) * vec( A(θ) * U Λ^{1/2} )   -> 4×1 列向量
d_beta_vec = @(theta_deg, beta) reshape( sqrt(N) * (A_mat(theta_deg) * U_sqrtLambda(beta)), [], 1);

% d_β(θ) 对角度（弧度）的导数 -> 4×1 列向量
d_dbeta_dtheta_vec = @(theta_deg, beta) reshape( sqrt(N) * (dA_dtheta(theta_deg) * U_sqrtLambda(beta)), [], 1);

%% 计算 CRB 的主函数
for k = 1:length(theta2_deg_list)
    theta2_deg = theta2_deg_list(k);
    for b_idx = 1:length(beta_vals)
        beta = beta_vals(b_idx);
        
        % 等效导向矢量及其导数 (均为 4×1)
        d1 = d_beta_vec(theta1_deg, beta);
        d2 = d_beta_vec(theta2_deg, beta);
        d1p = d_dbeta_dtheta_vec(theta1_deg, beta);
        d2p = d_dbeta_dtheta_vec(theta2_deg, beta);
        
        % 构造梯度矩阵 ∂μ/∂ξ (4个观测×6个实参数)
        % 参数顺序: θ1, θ2, Re(α1), Im(α1), Re(α2), Im(α2)
        grad = zeros(length(d1), 6);
        grad(:,1) = alpha1 * d1p;
        grad(:,2) = alpha2 * d2p;
        grad(:,3) = d1;
        grad(:,4) = 1j * d1;
        grad(:,5) = d2;
        grad(:,6) = 1j * d2;
        
        % Fisher信息矩阵 J = (2/σ_w^2) * Re( grad^H * grad )
        J = (2/sigma_w2) * real(grad' * grad);
        
        % 检查条件数，避免奇异
        if rcond(J) < 1e-12
            CRB_theta1_deg = Inf;
        else
            J_inv = inv(J);
            CRB_var_rad2 = J_inv(1,1);   % θ1的方差（弧度²）
            CRB_theta1_deg = sqrt(CRB_var_rad2) * (180/pi);  % 标准差（度）
        end
        CRB_results(k, b_idx) = CRB_theta1_deg;
    end
end

%% 绘图
figure(1);

colors = {'b', 'r', 'g'};
for k = 1:length(theta2_deg_list)
    semilogy(beta_vals, CRB_results(k,:), 'Color', colors{k}, 'LineWidth', 2, 'DisplayName', sprintf('\\theta_2 = %d°', theta2_deg_list(k)));
    hold on;
 end
xlabel('Correlation coefficient \beta');
ylabel('CRB on DOA (deg)');
title('Figure 8: CRB on DOA estimation (M=2, L=2, SNR=0dB, \theta_1=0°)');
grid on; 
grid minor;
legend('Location', 'best');
% ylim([0, 1.2]);
% xlim([0, 1]);
hold off;

































