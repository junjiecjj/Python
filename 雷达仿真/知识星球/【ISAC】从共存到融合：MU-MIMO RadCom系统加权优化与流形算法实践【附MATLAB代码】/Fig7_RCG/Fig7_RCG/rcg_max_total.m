function T = rcg_max_total(H, R, Gamma_lin, P0, N0, rho, eps_max, opts)
% RCG_MAX_TOTAL  Riemannian conjugate gradient for max SINR penalty.
%
%   T = rcg_max_total(H, R, Gamma_lin, P0, N0, rho, eps_max, opts)
%
%   Solves the weighted penalty optimization with a max penalty on the
%   SINR constraints using the log‑sum‑exp approximation.  This
%   corresponds to Eqs. (33)–(42) in the paper.  The variable T lies
%   on the hypersphere manifold defined by ||T||_F^2 = P0.  The
%   algorithm uses Riemannian conjugate gradient with Armijo line search.
%
%   Inputs:
%     H        – N×K channel matrix
%     R        – N×N desired radar covariance
%     Gamma_lin– target SINR (linear)
%     P0       – total transmit power
%     N0       – noise power
%     rho      – 1×2 vector [rho1, rho2] weighting radar and SINR terms
%     eps_max  – smoothing parameter for log‑sum‑exp approximation
%     opts     – struct with fields maxIter, tolGrad, verbose, stepInit
%
%   Output:
%     T – N×K matrix of transmit beamformers

[N, K] = size(H);
rho1 = rho(1);
rho2 = rho(2);

% Extract algorithm parameters
maxIter  = opts.maxIter;
tolGrad  = opts.tolGrad;
verbose  = opts.verbose;
stepInit = opts.stepInit;

% Random initialization on the hypersphere
T = (randn(N,K) + 1j*randn(N,K))/sqrt(2);
T = sqrt(P0) * T / norm(T, 'fro');

% Initial gradient and search direction
[gradR, ~, ~] = grad_max_total(T, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
dir = -gradR;

for it = 1:maxIter
    gradNorm = norm(gradR, 'fro');
    if gradNorm < tolGrad
        if verbose
            fprintf('Max RCG converged at iter %d, ||grad||=%.3e\n', it, gradNorm);
        end
        break;
    end
    
    % Armijo line search
    step = stepInit;
    f_curr = obj_max_total(T, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
    c    = 1e-4;
    beta = 0.5;
    inner_grad_dir = real_trace(gradR, dir);
    
    while true
        T_trial = retraction_sphere(T, dir, step, P0);
        f_trial = obj_max_total(T_trial, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
        if f_trial <= f_curr + c * step * inner_grad_dir
            break;
        end
        step = step * beta;
        if step < 1e-6
            break;
        end
    end
    
    % Retraction step
    T_new = retraction_sphere(T, dir, step, P0);
    
    % New gradient
    [gradR_new, ~, ~] = grad_max_total(T_new, H, R, Gamma_lin, N0, rho1, rho2, eps_max);
    
    % Polak‑Ribière coefficient
    gradR_old_trans = proj_tangent_sphere(T_new, gradR);
    dir_old_trans   = proj_tangent_sphere(T_new, dir);
    num = real_trace(gradR_new, gradR_new - gradR_old_trans);
    den = real_trace(gradR, gradR);
    mu  = max(num/den, 0);
    
    % Update search direction
    dir_new = -gradR_new + mu * dir_old_trans;
    
    % Move to new iterate
    T     = T_new;
    gradR = gradR_new;
    dir   = dir_new;
end

end

%% -------------------------------------------------------------------------
function f = obj_max_total(T, H, R, Gamma, N0, rho1, rho2, eps_max)
% Objective function for the Max SINR penalty (log‑sum‑exp approximation).

[N, K] = size(H); %#ok<NASGU>
C = T * T';

% Radar mismatch term
term_radar = norm(C - R, 'fro')^2;

% Compute alpha_i and log‑sum‑exp penalty
alpha = zeros(K,1);
for i = 1:K
    hi  = H(:,i);
    ti  = T(:,i);
    hiTti = hi.' * ti;
    term1 = (1 + Gamma) * abs(hiTti)^2;
    hiT   = hi.' * T;
    term2 = Gamma * (hiT * hiT');
    alpha(i) = term1 - term2;
end

l_hat = eps_max * log(sum(exp(-alpha / eps_max)));

f = rho1 * term_radar + rho2 * l_hat;
end

%% -------------------------------------------------------------------------
function [gradR, alpha, weights] = grad_max_total(T, H, R, Gamma, N0, rho1, rho2, eps_max)
% Compute the Riemannian gradient for the Max SINR penalty.

[N, K] = size(H);
C = T * T';

% Radar term gradient
gradE = 4 * rho1 * (C - R) * T;

% Compute alpha_i and G_i for each user
alpha   = zeros(K,1);
G_list  = zeros(N, K, K);
for i = 1:K
    hi  = H(:,i);
    ti  = T(:,i);
    hiTti = hi.' * ti;
    term1 = (1 + Gamma) * abs(hiTti)^2;
    hiT   = hi.' * T;
    term2 = Gamma * (hiT * hiT');
    alpha(i) = term1 - term2;
    
    % Compute G_i = B_i * [(1+Gamma) t_i e_i^T − Gamma*T]
    ei   = zeros(K,1); ei(i) = 1;
    tmp  = (1 + Gamma) * ti * ei.' - Gamma * T;
    G_i  = conj(hi) * (hi.' * tmp);
    G_list(:,:,i) = G_i;
end

% Compute softmax‑like weights
w = exp(-alpha / eps_max);
w = w / sum(w);
weights = w; %#ok<NASGU>

% Weighted sum of G_i
G_weighted = zeros(N, K);
for i = 1:K
    G_weighted = G_weighted + w(i) * G_list(:,:,i);
end

% Gradient of the SINR penalty
gradE = gradE - 2 * rho2 * G_weighted;

% Project onto tangent space
gradR = proj_tangent_sphere(T, gradE);
end