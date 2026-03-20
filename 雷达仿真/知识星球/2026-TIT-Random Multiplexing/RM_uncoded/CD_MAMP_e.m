%% Cross-domain MAMP (CD-MAMP)
% Model: 
% y = H * U * s + n, n ~ CN(0, v_n)
% U is a unitary modulation matrix
% **size(H) should be large (typically >500, maybe okay for 100~500)**
% ------------------------------------------------------------------
% Input:
% (1) H: channel matrix  
% (2) s, signal s (used only when computing MSE)
%     y, v_n: received signal y, noise variance v_n
% (3) L: damping length, typically L = 3
% (4) it: maximum number of iterations
% (5) info: prior information of s (e.g. PSK, QAM, BG)
% (6) type: string, see "random_transform.m"
% (7) index_p: vector, permutation Pi
% ------------------------------------------------------------------
% Output:
% (1) MSE, Var: MSE and variance of s_post in iterations (help debug)
% (2) s_post: soft estimate of s
% ------------------------------------------------------------------
function [MSE, Var, s_post] = CD_MAMP_e(H, s, y, v_n, L, it, info, mod_info)
    [M, N] = size(H);
    delta = M / N;
    % estimate lam_s of A*A^H
    L1 = ceil(1000/N);
    tau = 20;
    lam_s = est_lams(H, tau, L1);
    % estimate chi_0, chi_1
    L2 = ceil(1000/N);
    theta_1 = 1 / (lam_s + v_n / info.var);     % theta_0 in "Overflow-Avoiding MAMP"
    h = zeros(M, it);
    chi = zeros(1, 2*it-1);
    [w_0, chi(1), h] = est_chi(H, h, lam_s, theta_1, 1, L2);      % w_0 = chi_0
    w_1 = chi(1) / theta_1;
    wb_00 = lam_s * w_0 - w_1 - w_0 * w_0;
    %
    x_phi = zeros(N, it);
    v_phi = zeros(it, it);
    log_vth = zeros(1, it);                      % log(|\vartheta_{t,i}|)
    sgn_xi = zeros(1, it);                       % sgn(xi)               
    z = zeros(M, it);
    s_orth = info.mean * ones(N, 1);
    x_phi(:, 1) = Modulations(s_orth, mod_info, 0);   % cross domain
    z(:, 1) = y - H * x_phi(:, 1);
    v_phi(1, 1) = real(z(:, 1)' * z(:, 1) / N - delta * v_n) / w_0;
    log_th1 = log(theta_1);
    z_hat = zeros(M, 1);
    r_hat = zeros(N, 1);
    MSE = zeros(1, it);
    Var = zeros(1, it);
    thres_0 = 1e-7;                             
    
    % iterations
    for t = 1 : it
        % MLE
        [log_vth, sgn_xi, z_hat, r_hat, r, v_gam] = MLE_MAMP(H, x_phi, v_phi, chi, sgn_xi, log_vth, ...
        y, z_hat, r_hat, w_0, wb_00, lam_s, v_n, log_th1, t);
        % cross domain
        r = Modulations(r, mod_info, 1);
        % NLE
        [s_post, v_post] = Denoiser(r, v_gam, info);
        MSE(t) = (s_post - s)' * (s_post - s) / N;
        Var(t) = v_post;
        if v_post < thres_0             % Possible to add extra stopping conditions 
            MSE(t:end) = max(MSE(t), thres_0);
            Var(t:end) = max(Var(t), thres_0);
            break
        end
        s_orth = (s_post / v_post - r / v_gam) / (1 / v_post - 1 / v_gam);
        % cross domain
        x_phi(:, t+1) = Modulations(s_orth, mod_info, 0);
        % 
        z(:, t+1) = y - H * x_phi(:, t+1);
        v_phi(t+1, t+1) = (z(:, t+1)' * z(:, t+1) / N - delta * v_n) / w_0;
        for k = 1 : t
            v_phi(t+1, k) = (z(:, t+1)' * z(:, k) / N - delta * v_n) / w_0;
            v_phi(k, t+1) = v_phi(t+1, k)';
        end
        if v_phi(t+1, t+1) < 0
            v_phi(t+1, t+1) = 1 / (1 / v_post - 1 / v_gam);
        end
        % damping at NLE
        [x_phi, z, v_phi] = Opt_Damping(x_phi, v_phi, z, L, t+1);
        % estimate chi_{2t-2}, chi_{2t-1} (t <-- t+1 later)
        [chi(2*t), chi(2*t+1), h] = est_chi(H, h, lam_s, theta_1, t+1, L2);
    end
end

%% Optimized damping
% ------------------------------------------------------------------
% Input:
% (1) X: [x'_1, ..., x'_{tt-1}, x_{tt}], x_tt is undamped 
% (2) V: error covariance matrix for X
% (3) Z: [z'_1, ..., z'_{tt-1}, z_{tt}], z_i = y - H * x_i
% (4) L: damping length, typically L = 3
% (5) tt: tt = t+1 if damping at NLE, otherwise tt = t
% ------------------------------------------------------------------
% Output:
% (1) X: [x'_1, ..., x'_{tt-1}, x'_{tt}] (x_{tt} -> x'_{tt})
% (2) V: error covariance matrix for new X
% (3) Z: [z'_1, ..., z'_{tt-1}, z'_{tt}] (z_{tt} -> z'_{tt})
% ------------------------------------------------------------------
function [X, Z, V] = Opt_Damping(X, V, Z, L, tt)
    ll = min(L, tt);
    % Tikhonov regularization (for robustness)
    alpha = 1e-12;
    V_da = V(tt-ll+1:tt, tt-ll+1:tt) + alpha * eye(ll);
    % Damping vector
    tmp = V_da \ ones(ll, 1);
    v_s = real(ones(1, ll) * tmp);
    zeta = tmp / v_s;
    zeta = zeta.';
    % Update
    V(tt, tt) = 1 / v_s;
    X(:, tt) = sum(zeta.*X(:, tt-ll+1:tt), 2);
    Z(:, tt) = sum(zeta.*Z(:, tt-ll+1:tt), 2);
    for k = 1 : tt-1
        V(k, tt) = sum(zeta.*V(k, tt-ll+1:tt));
        V(tt, k) = V(k, tt)';
    end
end

%% Estimate average of largest and smallest eigenvalues of A*A^H
% Eigenvalues of A*A^H is the same as those of H*H^H
% ------------------------------------------------------------------
% Input:
% (1) H: sparse time-domain channel matrix 
% (2) tau: typically 20~30, lam_s becomes tighter as tau increases
% (3) L_1: typically =1 or ceil(1000/N)
% ------------------------------------------------------------------
% Output:
% lam_s: estimate of (lambda_max + lambda_min) / 2 
% ------------------------------------------------------------------
function lam_s = est_lams(H, tau, L1)
    lam = 0;
    N = size(H, 2);
    for k = 1 : L1
        s0_re = normrnd(0, 1, [N, 1]);
        s0_im = normrnd(0, 1, [N, 1]);
        s0 = s0_re + s0_im*1i;
        s0 = s0 / norm(s0);
        s_i = s0;
        for i = 1 : tau
            if mod(i,2) == 1
                s_i = H * s_i;
            else
                s_i = H' * s_i;
            end
            tmp = real(s_i' * s_i);
        end
        lam = lam + tmp;
    end
    lam = lam / L1;                             
    lam_s = 0.5 * (N * lam)^(1/tau);
end

%% chi_k = tr{A^H*B^k*A} = tr{A^H*(lam_s*I - A*A^H)^k*A}, k \in {2t-2, 2t-1}
% Replace A with H does not change the eigenvalues!
% See the paper "Overflow-Avoiding Memory AMP" for details
% ------------------------------------------------------------------
% Input:
% (1) H: sparse time-domain channel matrix 
% (2) h: matrix, h_j = theta_1 * (lam_s * h_{j-1} - H * H' * h_{j-1})
%        h0 ~ CN(0, 1)
% (3) lam_s: given by "est_lams"
% (4) theta_1: parameter
% (5) L_2: typically =1 or ceil(1000/N)
% ------------------------------------------------------------------
% Output:
% (1) chi_0, chi_1: chi_{2t-2}, chi_{2t-1} in the paper
% (2) h: updated h
% ------------------------------------------------------------------
function [chi_0, chi_1, h] = est_chi(H, h, lam_s, theta_1, t, L2)
    chi_0 = 0;
    chi_1 = 0;
    if t == 1
        N = size(H, 2);
        for k = 1 : L2
            h0_re = normrnd(0, 1, [N, 1]);
            h0_im = normrnd(0, 1, [N, 1]);
            h0 = h0_re + h0_im*1i;
            h0 = H * (h0 / norm(h0));
            h(:, 1) = theta_1 * (lam_s * h0 - H * (H' * h0));
            chi_0 = chi_0 + h0' * h0;
            chi_1 = chi_1 + real(h(:, 1)' * h0);
        end
        chi_0 = chi_0 / L2;
        chi_1 = chi_1 / L2;
    else
        for k = 1 : L2
            h(:, t) = theta_1 * (lam_s * h(:, t-1) - H * (H' * h(:, t-1)));
            chi_0 = chi_0 + h(:, t-1)' * h(:, t-1);
            chi_1 = chi_1 + real(h(:, t-1)' * h(:, t));
        end
        chi_0 = chi_0 / L2;
        chi_1 = chi_1 / L2;
    end
end