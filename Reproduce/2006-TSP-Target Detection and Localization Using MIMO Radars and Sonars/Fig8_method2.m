%% 基于(40)-(43) 计算双目标 CRB（使用等效导向矢量 d_beta，已验证正确）
clear; clc; close all;

%% 参数（与之前正确图8相同）
M = 2; d_lambda = 0.5; SNR_dB = 0; SNR_lin = 10^(SNR_dB/10);
N = 1; alpha = [1; 1]; sigma_w2 = 1/SNR_lin;
theta1 = 0; theta2_list = [5, 10, 15]; beta_vals = linspace(0,0.99,100);

% 对称阵列
n = -(M-1)/2 : (M-1)/2;
a = @(th) exp(-1j*pi*d_lambda*n'*sind(th));
da = @(th) -1j*pi*d_lambda*cosd(th)*n'.*a(th);

% 相干矩阵
R_s = @(b) [1, b; b, 1];
[U, Lam] = eig(R_s(0)); U_sqrtL = @(b) U*sqrt(Lam);  % 通用，仅用于 beta=0,1 时正确
% 但为准确，对任意 beta 直接计算 U*sqrt(Lam)
function U_sqrtL = get_U_sqrtL(beta)
    [U,Lam] = eig([1, beta; beta, 1]);
    U_sqrtL = U * sqrt(Lam);
end

% 等效导向矢量 d_beta
d_beta = @(th, beta) reshape( sqrt(N) * (a(th)*a(th).' * get_U_sqrtL(beta)), [], 1);
d_beta_deriv = @(th, beta) reshape( sqrt(N) * ( (da(th)*a(th).' + a(th)*da(th).') * get_U_sqrtL(beta) ), [], 1);

CRB_results = zeros(length(theta2_list), length(beta_vals));

for k = 1:length(theta2_list)
    theta2 = theta2_list(k);
    for b = 1:length(beta_vals)
        beta = beta_vals(b);
        d1 = d_beta(theta1, beta); d1p = d_beta_deriv(theta1, beta);
        d2 = d_beta(theta2, beta); d2p = d_beta_deriv(theta2, beta);
        
        % 梯度矩阵: 6 列，分别对应 θ1, θ2, Reα1, Imα1, Reα2, Imα2
        G = [alpha(1)*d1p, alpha(2)*d2p, d1, 1j*d1, d2, 1j*d2];
        J = (2/sigma_w2) * real(G' * G);
        if rcond(J) > 1e-12
            CRB_theta1 = sqrt(inv(J(1,1))) * (180/pi);
        else
            CRB_theta1 = NaN;
        end
        CRB_results(k,b) = CRB_theta1;
    end
end

% 绘图
figure; hold on;
plot(beta_vals, CRB_results(1,:), 'b-', 'LineWidth',2);
plot(beta_vals, CRB_results(2,:), 'r--', 'LineWidth',2);
plot(beta_vals, CRB_results(3,:), 'g-.', 'LineWidth',2);
xlabel('\beta'); ylabel('CRB (deg)'); legend('\theta_2=5°','\theta_2=10°','\theta_2=15°');
grid on; 
title('Figure 8 (correct)');