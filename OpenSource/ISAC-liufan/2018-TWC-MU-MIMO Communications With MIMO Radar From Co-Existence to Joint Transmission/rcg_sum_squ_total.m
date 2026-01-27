function T = rcg_sum_squ_total(H, R, Gamma_lin, P0, N0, rho, opts)
% ------------------------------------------------------------------
% RCG for Sum-Square SINR penalty under Total Power constraint
%
% Minimize:
%   f(T) = rho1 * || T T^H - R ||_F^2 + rho2 * sum_i (alpha_i - N0*Gamma)^2
%
% where
%   alpha_i = (1+Gamma)*|h_i^T t_i|^2 - Gamma * || T^H h_i^* ||^2
%
% Constraint:
%   ||T||_F^2 = P0  (complex hypersphere manifold)
%
% H: (N x K), columns are h_i
% R: (N x N)
% T: (N x K)
% ------------------------------------------------------------------

[N, K] = size(H);
rho1 = rho(1);
rho2 = rho(2);

maxIter  = opts.maxIter;
tolGrad  = opts.tolGrad;
verbose  = opts.verbose;
stepInit = opts.stepInit;

% 初始化 T (随机复数, 归一化到 Fro norm = sqrt(P0))
T = (randn(N,K) + 1j*randn(N,K))/sqrt(2);
T = sqrt(P0) * T / norm(T, 'fro');

% 初始梯度 & 方向
[gradR, ~] = grad_sum_squ_total(T, H, R, Gamma_lin, N0, rho1, rho2);
dir = -gradR;

for it = 1:maxIter
    
    gradNorm = norm(gradR, 'fro');
    if gradNorm < tolGrad
        if verbose
            fprintf('Sum-Squ-RCG converged at iter %d, ||grad||=%g\n', it, gradNorm);
        end
        break;
    end
    
    % ----- Armijo backtracking line search -----
    step = stepInit;
    f_curr = obj_sum_squ_total(T, H, R, Gamma_lin, N0, rho1, rho2);
    c = 1e-4; beta = 0.5;
    inner_grad_dir = real_trace(gradR, dir);
    
    while true
        T_trial = retraction_sphere(T, dir, step, P0);
        f_trial = obj_sum_squ_total(T_trial, H, R, Gamma_lin, N0, rho1, rho2);
        if f_trial <= f_curr + c*step*inner_grad_dir
            break;
        end
        step = step*beta;
        if step < 1e-6
            break;
        end
    end
    
    % ----- 更新 T -----
    T_new = retraction_sphere(T, dir, step, P0);
    
    % ----- 新梯度 -----
    [gradR_new, ~] = grad_sum_squ_total(T_new, H, R, Gamma_lin, N0, rho1, rho2);
    
    % ----- Riemannian Polak-Ribiere 系数 μ (式(50)) -----
    % 向量搬运 (vector transport): 对旧梯度和方向做投影到新切空间
    gradR_old_trans = proj_tangent_sphere(T_new, gradR);
    dir_old_trans   = proj_tangent_sphere(T_new, dir);
    
    num = real_trace(gradR_new, gradR_new - gradR_old_trans);
    den = real_trace(gradR, gradR);
    mu  = max(num/den, 0);     % 可选：保证下降方向
    
    % ----- 新方向 (式(51)) -----
    dir_new = -gradR_new + mu * dir_old_trans;
    
    % 更新
    T     = T_new;
    gradR = gradR_new;
    dir   = dir_new;
end

end

%% --------- 目标函数 f(T) for Sum-Squ --------------------------
function f = obj_sum_squ_total(T, H, R, Gamma, N0, rho1, rho2)
[N, K] = size(H);
C = T*T';       % covariance

% Radar term
term_radar = norm(C - R, 'fro')^2;

% SINR penalty term
alpha = zeros(K,1);
for i = 1:K
    hi  = H(:,i);
    ti  = T(:,i);
    hiTti = hi.'*ti;
    term1 = (1+Gamma)*abs(hiTti)^2;
    hiT   = hi.'*T;              % 1 x K
    term2 = Gamma*(hiT*hiT');    % scalar
    alpha(i) = term1 - term2;
end
lambda = sum( (alpha - N0*Gamma).^2 );

f = rho1*term_radar + rho2*lambda;
end

%% --------- Riemannian gradient for Sum-Squ ---------------------
function [gradR, alpha] = grad_sum_squ_total(T, H, R, Gamma, N0, rho1, rho2)
[N, K] = size(H);

% Euclidean gradient
C = T*T';   % covariance

% Radar part
gradE = 4*rho1*(C - R)*T;

% SINR penalty part
alpha = zeros(K,1);
Gsum  = zeros(N,K);
for i = 1:K
    hi  = H(:,i);
    ti  = T(:,i);
    hiTti = hi.'*ti;
    term1 = (1+Gamma)*abs(hiTti)^2;
    hiT   = hi.'*T;
    term2 = Gamma*(hiT*hiT');
    alpha(i) = term1 - term2;
    
    % G_i = B_i[(1+Gamma) t_i e_i^T - Gamma T]
    ei   = zeros(K,1); ei(i) = 1;
    tmp  = (1+Gamma)*ti*ei.' - Gamma*T;     % N x K
    % B_i = h_i^* h_i^T => B_i * tmp = h_i^* (h_i^T tmp)
    G_i  = (conj(hi)*(hi.'*tmp));
    
    Gsum = Gsum + alpha(i)*G_i;
end

gradE = gradE + 4*rho2*Gsum;

% Riemannian gradient on hypersphere (式(39))
gradR = proj_tangent_sphere(T, gradE);

end
