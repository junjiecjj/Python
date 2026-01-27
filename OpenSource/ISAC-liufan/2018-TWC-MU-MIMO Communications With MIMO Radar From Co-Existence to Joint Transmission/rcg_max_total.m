function T = rcg_max_total(H, R, Gamma_lin, P0, N0, rho, eps_max, opts)
% ------------------------------------------------------------------
% RCG for Max SINR penalty (log-sum-exp) under Total Power constraint
%
% Minimize:
%   f(T) = rho1 * || T T^H - R ||_F^2 + rho2 * \hat{l}(\alpha)
%
% where
%   \hat{l}(\alpha) = eps * log( sum_i exp(-alpha_i/eps) )
%   alpha_i same as in Sum-Squ case
% ------------------------------------------------------------------

[N, K] = size(H);
rho1 = rho(1);
rho2 = rho(2);

maxIter  = opts.maxIter;
tolGrad  = opts.tolGrad;
verbose  = opts.verbose;
stepInit = opts.stepInit;

% 初始化 T
T = (randn(N,K) + 1j*randn(N,K))/sqrt(2);
T = sqrt(P0) * T / norm(T, 'fro');

[gradR, ~, ~] = grad_max_total(T, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
dir = -gradR;

for it = 1:maxIter
    
    gradNorm = norm(gradR,'fro');
    if gradNorm < tolGrad
        if verbose
            fprintf('Max-RCG converged at iter %d, ||grad||=%g\n', it, gradNorm);
        end
        break;
    end
    
    % ----- Armijo line search -----
    step = stepInit;
    f_curr = obj_max_total(T, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
    c = 1e-4; beta = 0.5;
    inner_grad_dir = real_trace(gradR, dir);
    
    while true
        T_trial = retraction_sphere(T, dir, step, P0);
        f_trial = obj_max_total(T_trial, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
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
    [gradR_new, ~, ~] = grad_max_total(T_new, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
    
    % ----- Polak-Ribiere 系数 μ -----
    gradR_old_trans = proj_tangent_sphere(T_new, gradR);
    dir_old_trans   = proj_tangent_sphere(T_new, dir);
    
    num = real_trace(gradR_new, gradR_new - gradR_old_trans);
    den = real_trace(gradR, gradR);
    mu  = max(num/den, 0);
    
    % ----- 新方向 -----
    dir_new = -gradR_new + mu*dir_old_trans;
    
    T     = T_new;
    gradR = gradR_new;
    dir   = dir_new;
end

end

%% --------- 目标函数 for Max penalty ---------------------------
function f = obj_max_total(T, H, R, Gamma, N0, rho1, rho2, eps_max)
[N, K] = size(H);
C = T*T';

% Radar term
term_radar = norm(C-R,'fro')^2;

alpha = zeros(K,1);
for i = 1:K
    hi  = H(:,i);
    ti  = T(:,i);
    hiTti = hi.'*ti;
    term1 = (1+Gamma)*abs(hiTti)^2;
    hiT   = hi.'*T;
    term2 = Gamma*(hiT*hiT');
    alpha(i) = term1 - term2;
end

% log-sum-exp approximation of max(-alpha_i)
% l_hat = eps * log( sum exp(-alpha_i/eps) )
l_hat = eps_max * log( sum( exp(-alpha/eps_max) ) );

f = rho1*term_radar + rho2*l_hat;
end

%% --------- Riemannian gradient for Max penalty -----------------
function [gradR, alpha, weights] = grad_max_total(T, H, R, Gamma, N0, rho1, rho2, eps_max)
[N, K] = size(H);
C = T*T';

% Radar part
gradE = 4*rho1*(C - R)*T;

% alpha_i & G_i
alpha  = zeros(K,1);
G_list = zeros(N,K,K);  % store each G_i
for i = 1:K
    hi  = H(:,i);
    ti  = T(:,i);
    hiTti = hi.'*ti;
    term1 = (1+Gamma)*abs(hiTti)^2;
    hiT   = hi.'*T;
    term2 = Gamma*(hiT*hiT');
    alpha(i) = term1 - term2;
    
    ei   = zeros(K,1); ei(i) = 1;
    tmp  = (1+Gamma)*ti*ei.' - Gamma*T;
    G_i  = (conj(hi)*(hi.'*tmp));
    G_list(:,:,i) = G_i;
end

w = exp(-alpha/eps_max);
w = w / sum(w);        % 归一化权重
weights = w;

% \sum w_i * G_i
G_weighted = zeros(N,K);
for i = 1:K
    G_weighted = G_weighted + w(i)*G_list(:,:,i);
end

gradE = gradE - 2*rho2*G_weighted;

% Riemannian projection
gradR = proj_tangent_sphere(T, gradE);

end
