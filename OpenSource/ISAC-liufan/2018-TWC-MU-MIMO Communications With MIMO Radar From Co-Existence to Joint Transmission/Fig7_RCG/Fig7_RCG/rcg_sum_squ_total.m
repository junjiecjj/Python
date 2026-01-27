function T = rcg_sum_squ_total(H, R, Gamma_lin, P0, N0, rho, opts)
% RCG_SUM_SQU_TOTAL  Riemannian conjugate gradient for Sum‑Squ penalty.
%
%   T = rcg_sum_squ_total(H, R, Gamma_lin, P0, N0, rho, opts)
%
%   Solves the weighted penalty optimization problem (Eq. (31) in the
%   paper) on the hypersphere manifold (total power constraint).  The
%   objective is
%       rho1 * ||T T^H − R||_F^2 + rho2 * sum_i (alpha_i − N0*Gamma)^2,
%   where alpha_i is defined in Eq. (24).  The gradient is projected
%   onto the tangent space of the hypersphere.  A simple Armijo
%   backtracking line search is used.
%
%   Inputs:
%     H      – N×K channel matrix (columns are user channels h_i)
%     R      – N×N target radar covariance matrix
%     Gamma_lin – target SINR (linear scale)
%     P0     – total transmit power (scalar)
%     N0     – noise power (scalar)
%     rho    – 1×2 vector [rho1, rho2] weighting radar and SINR terms
%     opts   – struct with fields maxIter, tolGrad, verbose, stepInit
%
%   Output:
%     T – N×K matrix of transmit beamformers (columns are t_i)

[N, K] = size(H);
rho1 = rho(1);
rho2 = rho(2);

% Extract RCG parameters
maxIter  = opts.maxIter;
tolGrad  = opts.tolGrad;
verbose  = opts.verbose;
stepInit = opts.stepInit;

% Initialize T uniformly on the hypersphere ||T||_F^2 = P0
T = (randn(N,K) + 1j*randn(N,K))/sqrt(2);
T = sqrt(P0) * T / norm(T,'fro');

% Compute initial Riemannian gradient and direction
[gradR, ~] = grad_sum_squ_total(T, H, R, Gamma_lin, N0, rho1, rho2);
dir = -gradR;

for it = 1:maxIter
    gradNorm = norm(gradR,'fro');
    if gradNorm < tolGrad
        if verbose
            fprintf('Sum‑Squ RCG converged at iter %d, ||grad||=%.3e\n', it, gradNorm);
        end
        break;
    end
    
    % Armijo backtracking line search
    step = stepInit;
    f_curr = obj_sum_squ_total(T, H, R, Gamma_lin, N0, rho1, rho2);
    c    = 1e-4;
    beta = 0.5;
    inner_grad_dir = real_trace(gradR, dir);

    while true
        T_trial = retraction_sphere(T, dir, step, P0);
        f_trial = obj_sum_squ_total(T_trial, H, R, Gamma_lin, N0, rho1, rho2);
        if f_trial <= f_curr + c * step * inner_grad_dir
            break;
        end
        step = step * beta;
        if step < 1e-6
            break;
        end
    end
    
    % Retract to new point on the manifold
    T_new = retraction_sphere(T, dir, step, P0);
    
    % Compute new gradient
    [gradR_new, ~] = grad_sum_squ_total(T_new, H, R, Gamma_lin, N0, rho1, rho2);
    
    % Polak‑Ribière coefficient µ (with transport)
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
function f = obj_sum_squ_total(T, H, R, Gamma, N0, rho1, rho2)
% Objective function for the Sum‑Squ penalty optimization.

[N, K] = size(H); %#ok<NASGU>

C = T * T';           % covariance matrix

% Radar mismatch term
term_radar = norm(C - R, 'fro')^2;

% SINR mismatch term
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
lambda = sum((alpha - N0 * Gamma).^2);

f = rho1 * term_radar + rho2 * lambda;
end

%% -------------------------------------------------------------------------
function [gradR, alpha] = grad_sum_squ_total(T, H, R, Gamma, N0, rho1, rho2)
% Compute the Riemannian gradient for the Sum‑Squ penalty.

[N, K] = size(H);

C = T * T';

% Radar term: Euclidean gradient of ||T T^H − R||_F^2
gradE = 4 * rho1 * (C - R) * T;

% SINR penalty term
alpha = zeros(K,1);
Gsum  = zeros(N, K);
for i = 1:K
    hi  = H(:,i);
    ti  = T(:,i);
    hiTti = hi.' * ti;
    term1 = (1 + Gamma) * abs(hiTti)^2;
    hiT   = hi.' * T;
    term2 = Gamma * (hiT * hiT');
    alpha(i) = term1 - term2;
    
    % G_i = B_i * [(1+Gamma)*t_i e_i^T − Gamma*T]
    ei   = zeros(K,1); ei(i) = 1;
    tmp  = (1 + Gamma) * ti * ei.' - Gamma * T;
    G_i  = conj(hi) * (hi.' * tmp);
    
    Gsum = Gsum + alpha(i) * G_i;
end

gradE = gradE + 4 * rho2 * Gsum;

% Project onto tangent space of the hypersphere
gradR = proj_tangent_sphere(T, gradE);
end