% =============================================================
%  Fig.7 - PSLR vs Average SINR (Shared deployment)
%  MU-MIMO Communications with MIMO Radar: From Co-existence
%  to Joint Transmission (Liu et al., TWC 2018)
%
%  Curves (6):
%   1) Constrained-SDR  (Total power)
%   2) Constrained-SDR  (Per-antenna)
%   3) Weighted-RCG Sum-Squ (Total power)   rho=[10,1]
%   4) Weighted-RCG Max     (Total power)   rho=[10,1]
%   5) Weighted-RCG Sum-Squ (Per-antenna)   rho=[3,1]
%   6) Weighted-RCG Max     (Per-antenna)   rho=[1,2]
%
%  System settings consistent with paper Fig.7:
%    N=20, K=10, P0=20dBm, N0=0dBm, ULA d=lambda/2
%  Radar beampattern target R: 3 dB beam centered at 0 deg, width 10 deg,
%    via Problem (10) under per-antenna diag(R)=P0/N.
%
%  Output:
%    Plot: PSLR (dB) vs Average SINR (dB)
% =============================================================
clear; clc; close all;

%% ---------------- Parameters (Fig.7) ----------------
P0_dBm = 20;      P0 = 10^(P0_dBm/10);     % linear
N0_dBm = 0;       N0 = 10^(N0_dBm/10);     % linear
N = 20;
K = 10;

lambda = 1;
d = 0.5*lambda;

rng(1);

% Angle grid for beampattern + PSLR
theta_deg = -90:0.5:90;
theta_rad = deg2rad(theta_deg);
M = numel(theta_deg);

A = zeros(N,M);
for m = 1:M
    A(:,m) = exp(1j*2*pi*d*(0:N-1)'*sin(theta_rad(m))/lambda);
end
S = cell(M,1);
for m = 1:M
    am = A(:,m);
    S{m} = am*am';
end

% 3dB mainlobe region: center 0 deg, width 10 deg => [-5, +5]
theta0 = 0; bw3 = 10;
th1 = theta0 - bw3/2; th2 = theta0 + bw3/2;
idx_main = find(theta_deg>=th1 & theta_deg<=th2);
idx_side = find(theta_deg<th1 | theta_deg>th2);

% Gamma sweep (Fig.7 x-axis shows avg SINR around 5~12 dB)
Gamma_dB_vec = 5:2:11;     % 7 points (matches the visual density of Fig.7)
Gnum = numel(Gamma_dB_vec);

% Monte Carlo (paper没在正文明确写次数；你可以改大以更平滑)
MC = 10;

% Table II weights for Fig.7
rho_total_sum  = [10, 1];
rho_total_max  = [10, 1];
rho_per_sum    = [ 3, 1];
rho_per_max    = [ 1, 2];

% log-sum-exp smoothing epsilon (paper: epsilon is "small positive")
eps_lse = 1e-1;

% RCG configs
rcg.maxit = 300;
rcg.tol   = 1e-6;

%% ---------------- Storage ----------------
% Each method returns: avgSINR(dB), PSLR(dB)
SINR_avg = struct();
PSLR_val = struct();

names = { ...
 'SDR_total','SDR_per', ...
 'RCG_sum_total','RCG_max_total', ...
 'RCG_sum_per','RCG_max_per'};

for i=1:numel(names)
    SINR_avg.(names{i}) = zeros(Gnum,1);
    PSLR_val.(names{i}) = zeros(Gnum,1);
end

%% =============================================================
%  Main Monte Carlo
% =============================================================
for ig = 1:Gnum
    Gamma_dB = Gamma_dB_vec(ig);
    Gamma = 10^(Gamma_dB/10) * ones(K,1);

    sinr_acc = cellfun(@(x)0, cell(size(names)));
    pslr_acc = cellfun(@(x)0, cell(size(names)));

    for mc = 1:MC
        % ----- Channels (Rayleigh): H is N x K, hi is i-th column -----
        H = (randn(N,K)+1j*randn(N,K))/sqrt(2);

        Bi = cell(K,1);
        for k = 1:K
            hk = H(:,k);
            Bi{k} = conj(hk)*transpose(hk);  % Bi = h_i^* h_i^T (paper)
        end

        % =========================================================
        %  Step 1) Radar-only target covariance R (Problem 10, 3dB)
        %    diag(R)=P0/N, maximize t s.t. 3dB + sidelobe margin
        % =========================================================
        R = radar_only_3db_CVX(S, theta_deg, theta0, bw3, idx_side, P0, N);

        % =========================================================
        %  (A) Constrained SDR (Total power): trace(Wsum)=P0
        % =========================================================
        [Wsum_t] = radcom_constrained_SDR_CVX(S, R, Bi, Gamma, N0, P0, 'total');
        T_t = rank1_from_cov(Wsum_t, K, P0, 'total');

        [sinr_db, pslr_db] = eval_sinr_pslr(T_t, H, A, idx_main, idx_side, N0);
        sinr_acc(1) = sinr_acc(1) + sinr_db;
        pslr_acc(1) = pslr_acc(1) + pslr_db;

        % =========================================================
        %  (B) Constrained SDR (Per-antenna): diag(Wsum)=P0/N
        % =========================================================
        [Wsum_p] = radcom_constrained_SDR_CVX(S, R, Bi, Gamma, N0, P0, 'per');
        T_p = rank1_from_cov(Wsum_p, K, P0, 'per');

        [sinr_db, pslr_db] = eval_sinr_pslr(T_p, H, A, idx_main, idx_side, N0);
        sinr_acc(2) = sinr_acc(2) + sinr_db;
        pslr_acc(2) = pslr_acc(2) + pslr_db;

        % =========================================================
        %  (C) Weighted RCG Sum-Squ (Total power)  f1: (32)+(23)
        % =========================================================
        T0 = init_T_random(N,K,P0);
        T_sum = rcg_hypersphere_sum(T0, R, Bi, Gamma, N0, rho_total_sum, rcg);
        [sinr_db, pslr_db] = eval_sinr_pslr(T_sum, H, A, idx_main, idx_side, N0);
        sinr_acc(3) = sinr_acc(3) + sinr_db;
        pslr_acc(3) = pslr_acc(3) + pslr_db;

        % =========================================================
        %  (D) Weighted RCG Max (Total power)  f2: (41)+(40)
        % =========================================================
        T0 = init_T_random(N,K,P0);
        T_max = rcg_hypersphere_max(T0, R, Bi, Gamma, N0, rho_total_max, eps_lse, rcg);
        [sinr_db, pslr_db] = eval_sinr_pslr(T_max, H, A, idx_main, idx_side, N0);
        sinr_acc(4) = sinr_acc(4) + sinr_db;
        pslr_acc(4) = pslr_acc(4) + pslr_db;

        % =========================================================
        %  (E) Weighted RCG Sum-Squ (Per-antenna) f3: (56)+(23) on oblique
        % =========================================================
        X0 = init_X_random(K,N,P0); % X = T^H, column norms fixed
        X_sum = rcg_oblique_sum(X0, R, H, Gamma, rho_per_sum, P0, eps_lse, rcg, N0);
        T_sum_per = X_sum.'; % back to T
        [sinr_db, pslr_db] = eval_sinr_pslr(T_sum_per, H, A, idx_main, idx_side, N0);
        sinr_acc(5) = sinr_acc(5) + sinr_db;
        pslr_acc(5) = pslr_acc(5) + pslr_db;

        % =========================================================
        %  (F) Weighted RCG Max (Per-antenna) f4: (57)+(40) on oblique
        % =========================================================
        X0 = init_X_random(K,N,P0);
        X_max = rcg_oblique_max(X0, R, H, Gamma, rho_per_max, P0, eps_lse, rcg, N0);
        T_max_per = X_max.';
        [sinr_db, pslr_db] = eval_sinr_pslr(T_max_per, H, A, idx_main, idx_side, N0);
        sinr_acc(6) = sinr_acc(6) + sinr_db;
        pslr_acc(6) = pslr_acc(6) + pslr_db;

        fprintf('Gamma=%2ddB, MC=%d/%d done.\n', Gamma_dB, mc, MC);
    end

    sinr_acc = sinr_acc / MC;
    pslr_acc = pslr_acc / MC;

    SINR_avg.SDR_total(ig)      = sinr_acc(1);
    PSLR_val.SDR_total(ig)      = pslr_acc(1);

    SINR_avg.SDR_per(ig)        = sinr_acc(2);
    PSLR_val.SDR_per(ig)        = pslr_acc(2);

    SINR_avg.RCG_sum_total(ig)  = sinr_acc(3);
    PSLR_val.RCG_sum_total(ig)  = pslr_acc(3);

    SINR_avg.RCG_max_total(ig)  = sinr_acc(4);
    PSLR_val.RCG_max_total(ig)  = pslr_acc(4);

    SINR_avg.RCG_sum_per(ig)    = sinr_acc(5);
    PSLR_val.RCG_sum_per(ig)    = pslr_acc(5);

    SINR_avg.RCG_max_per(ig)    = sinr_acc(6);
    PSLR_val.RCG_max_per(ig)    = pslr_acc(6);
end

%% ---------------- Plot Fig.7 style ----------------
figure; hold on; grid on;

plot(SINR_avg.SDR_total,     PSLR_val.SDR_total,     'ko-','LineWidth',1.6);
plot(SINR_avg.SDR_per,       PSLR_val.SDR_per,       'k--o','LineWidth',1.6);

plot(SINR_avg.RCG_sum_total, PSLR_val.RCG_sum_total, 'bs-','LineWidth',1.6);
plot(SINR_avg.RCG_sum_per,   PSLR_val.RCG_sum_per,   'b--s','LineWidth',1.6);

plot(SINR_avg.RCG_max_total, PSLR_val.RCG_max_total, 'rd-','LineWidth',1.6);
plot(SINR_avg.RCG_max_per,   PSLR_val.RCG_max_per,   'r--d','LineWidth',1.6);

xlabel('Average SINR (dB)');
ylabel('PSLR (dB)');
title('Fig.7  PSLR vs Average SINR (Shared Deployment)');
legend( ...
 'Constrained SDR (Total)', 'Constrained SDR (Per-antenna)', ...
 'Sum-Squ RCG (Total)',     'Sum-Squ RCG (Per-antenna)', ...
 'Max RCG (Total)',         'Max RCG (Per-antenna)', ...
 'Location','best');

%% =============================================================
%  -------------------- Subfunctions ---------------------------
% =============================================================

function R = radar_only_3db_CVX(S, theta_deg, theta0, bw3, idx_side, P0, N)
    th1 = theta0 - bw3/2; th2 = theta0 + bw3/2;
    [~, idx0] = min(abs(theta_deg-theta0));
    [~, idx1] = min(abs(theta_deg-th1));
    [~, idx2] = min(abs(theta_deg-th2));

    cvx_begin sdp quiet
        variable R(N,N) hermitian semidefinite
        variable t
        minimize(-t)
        subject to
            diag(R) == (P0/N)*ones(N,1);

            main0 = real(trace(R*S{idx0}));
            left  = real(trace(R*S{idx1}));
            right = real(trace(R*S{idx2}));
            left  == 0.5*main0;
            right == 0.5*main0;

            for mm = idx_side
                real(trace(R*S{mm})) <= main0 - t;
            end
    cvx_end
end

function Wsum_out = radcom_constrained_SDR_CVX(S, R, Bi, Gamma, N0, P0, mode)
    % Constrained SDR for Fig.7:
    %   minimize  sum_m ( tr(Wsum S_m) - sig * tr(R S_m) )^2
    %   s.t. SINR_i >= Gamma_i
    %        power constraint: trace(Wsum)=P0 (total) OR diag(Wsum)=P0/N (per)
    %
    K = numel(Bi);
    N = size(R,1);
    M = numel(S);

    % --- safety (since we hard-code W1..W10 below) ---
    if K ~= 10
        error('This implementation hard-codes K=10 (W1..W10). Current K=%d', K);
    end

    % Precompute radar reference pattern pr(m)=tr(R*S{m})
    pr = zeros(M,1);
    for m = 1:M
        pr(m) = real(trace(R * S{m}));
    end

    cvx_begin sdp quiet
        variable sig
        % user covariance matrices (K=10)
        variable W1(N,N) hermitian semidefinite
        variable W2(N,N) hermitian semidefinite
        variable W3(N,N) hermitian semidefinite
        variable W4(N,N) hermitian semidefinite
        variable W5(N,N) hermitian semidefinite
        variable W6(N,N) hermitian semidefinite
        variable W7(N,N) hermitian semidefinite
        variable W8(N,N) hermitian semidefinite
        variable W9(N,N) hermitian semidefinite
        variable W10(N,N) hermitian semidefinite

        % Wsum must be an expression (NOT a cvx variable)
        expression Wsum(N,N)
        Wsum = W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10;

        % beampattern matching residuals
        expression diff(M,1)
        for m = 1:M
            diff(m) = real(trace(Wsum * S{m})) - sig * pr(m);
        end

        minimize( sum_square(diff) )

        subject to
            sig >= 0;

            if strcmpi(mode,'total')
                trace(Wsum) == P0;
            else
                diag(Wsum) == (P0/N) * ones(N,1);
            end

            % SINR constraints:
            Ws = {W1,W2,W3,W4,W5,W6,W7,W8,W9,W10};
            for i = 1:K
                num = real(trace(Bi{i} * Ws{i}));
                den = 0;
                for k = 1:K
                    if k ~= i
                        den = den + real(trace(Bi{i} * Ws{k}));
                    end
                end
                num >= Gamma(i) * (den + N0);
            end
    cvx_end

    Wsum_out = Wsum;
end


function T = rank1_from_cov(Wsum, K, P0, mode)
    % Heuristic to extract rank-1 beamforming matrix T from covariance Wsum
    % (Fig.7 compares performance metrics; this extraction is standard in SDR usage)
    N = size(Wsum,1);
    [V,D] = eig((Wsum+Wsum')/2);
    [d,ix] = sort(real(diag(D)),'descend');
    V = V(:,ix); d = max(d,0);

    % Build K beams from top-K eigenmodes (simple & stable)
    kk = min(K, N);
    T = V(:,1:kk) * diag(sqrt(d(1:kk)));

    % If kk<K pad
    if kk < K
        T = [T, zeros(N, K-kk)];
    end

    % Enforce power constraint exactly
    if strcmpi(mode,'total')
        T = T * sqrt(P0 / (norm(T,'fro')^2 + eps));
    else
        % per-antenna: row scaling to make each antenna power = P0/N
        p = sum(abs(T).^2,2) + eps;
        scale = sqrt((P0/N)./p);
        T = diag(scale)*T;
    end
end

function T0 = init_T_random(N,K,P0)
    T0 = (randn(N,K)+1j*randn(N,K))/sqrt(2);
    T0 = T0 * (sqrt(P0)/(norm(T0,'fro')+eps));
end

function X0 = init_X_random(K,N,P0)
    % X is K x N, columns have norm sqrt(P0/N)
    X0 = (randn(K,N)+1j*randn(K,N))/sqrt(2);
    for n=1:N
        X0(:,n) = X0(:,n) / (norm(X0(:,n))+eps) * sqrt(P0/N);
    end
end

function [sinr_db, pslr_db] = eval_sinr_pslr(T, H, A, idx_main, idx_side, N0)
    % Average SINR
    K = size(H,2);
    sinr = zeros(K,1);
    for i=1:K
        hi = H(:,i);
       num = abs(hi' * T(:,i))^2;
        den = N0;
        for k=1:K
            if k~=i
                den = den + abs(hi' * T(:,k))^2;
            end
        end
        sinr(i) = num/(den+eps);
    end
    sinr_db = 10*log10(mean(sinr)+eps);

    % Beampattern and PSLR
    W = T*T';
    P = zeros(size(A,2),1);
    for m=1:size(A,2)
        am = A(:,m);
        P(m) = real(am' * W * am);
    end
    main_peak = max(P(idx_main));
    side_peak = max(P(idx_side));
    pslr_db = 10*log10( (main_peak+eps)/(side_peak+eps) );
end

%% ---------------- RCG on hypersphere (Total power) ----------------
function T = rcg_hypersphere_sum(T0, R, Bi, Gamma, N0, rho, cfg)
    % Min f1(T) = rho1||TT^H-R||_F^2 + rho2 * lambda(alpha)
    % lambda(alpha) = ||alpha - N0*Gamma||^2, alpha_i=(1+G_i)tr(Bi ti ti^H)-G_i tr(Bi TT^H)
    rho1=rho(1); rho2=rho(2);
    T = T0;
    g_prev = [];
    d_prev = [];

    for it=1:cfg.maxit
        [f, g] = f1_grad(T, R, Bi, Gamma, N0, rho1, rho2);    % Euclidean grad
        gR = proj_hyp(T, g);                                   % Riemannian grad

        if norm(gR,'fro') < cfg.tol, break; end

        if it==1
            d = -gR;
        else
            g_prev_tr = proj_hyp(T, g_prev);                   % vector transport
            beta = innerp(gR, gR - g_prev_tr) / (innerp(g_prev_tr,g_prev_tr)+eps);
            beta = max(beta,0);
            d = -gR + beta*d_prev;
        end

        % Armijo backtracking
        step = 1;
        c1 = 1e-4;
        for ls=1:30
            Tnew = retract_hyp(T, step*d);
            fnew = f1_only(Tnew, R, Bi, Gamma, N0, rho1, rho2);
            if fnew <= f + c1*step*innerp(gR,d), break; end
            step = step/2;
        end

        T = retract_hyp(T, step*d);
        if any(isnan(T(:))) || any(isinf(T(:)))
    warning('NaN detected in Max-RCG (Total), aborting this run');
    break;
        end

        g_prev = g;
        d_prev = d;
    end
end

function T = rcg_hypersphere_max(T0, R, Bi, Gamma, N0, rho, eps_lse, cfg)
    % Min f2(T) = rho1||TT^H-R||_F^2 + rho2 * l_hat(alpha)  (log-sum-exp)
    rho1=rho(1); rho2=rho(2);
    T = T0;
    g_prev = [];
    d_prev = [];

    for it=1:cfg.maxit
        [f, g] = f2_grad(T, R, Bi, Gamma, N0, rho1, rho2, eps_lse);
        gR = proj_hyp(T, g);

        if norm(gR,'fro') < cfg.tol, break; end

        if it==1
            d = -gR;
        else
            g_prev_tr = proj_hyp(T, g_prev);
            beta = innerp(gR, gR - g_prev_tr) / (innerp(g_prev_tr,g_prev_tr)+eps);
            beta = max(beta,0);
            d = -gR + beta*d_prev;
        end

        step = 1;
        c1 = 1e-4;
        for ls=1:30
            Tnew = retract_hyp(T, step*d);
            fnew = f2_only(Tnew, R, Bi, Gamma, N0, rho1, rho2, eps_lse);
            if fnew <= f + c1*step*innerp(gR,d), break; end
            step = step/2;
        end

        T = retract_hyp(T, step*d);
        g_prev = g;
        d_prev = d;
    end
end

function gR = proj_hyp(T, G)
    % grad f = G - Re(tr(T^H G))*T  (paper (39))
    gR = G - real(trace(T'*G))*T;
end

function Tnew = retract_hyp(T, Xi)
    % Tnew = sqrt(P0) * (T+Xi)/||T+Xi||_F, but sqrt(P0) already embedded in ||T||_F
    X = T + Xi;
    Tnew = X * (norm(T,'fro')/(norm(X,'fro')+eps));
end

function val = innerp(X,Y)
    val = real(trace(X'*Y));
end

function val = f1_only(T, R, Bi, Gamma, N0, rho1, rho2)
    W = T*T';
    e1 = norm(W-R,'fro')^2;
    alpha = alpha_total(T, Bi, Gamma);
    val = rho1*e1 + rho2*norm(alpha - N0*Gamma,'fro')^2;
end

function [f, G] = f1_grad(T, R, Bi, Gamma, N0, rho1, rho2)
    % paper (35): ∇f1 = 4rho1(WW-R)T + 4rho2 sum_i alpha_i G_i
    W = T*T';
    alpha = alpha_total(T, Bi, Gamma);
    f = rho1*norm(W-R,'fro')^2 + rho2*norm(alpha - N0*Gamma,'fro')^2;

    term1 = 4*rho1*(W-R)*T;

    K = size(T,2);
    term2 = zeros(size(T));
    for i=1:K
        ti = T(:,i);
        ei = zeros(K,1); ei(i)=1;
        Gi = Bi{i} * ( (1+Gamma(i))*ti*ei.' - Gamma(i)*T ); % paper (37)
        term2 = term2 + alpha(i)*Gi;
    end
    G = term1 + 4*rho2*term2;
end

function val = f2_only(T, R, Bi, Gamma, N0, rho1, rho2, eps_lse)
    W = T*T';
    e1 = norm(W-R,'fro')^2;
    alpha = alpha_total(T, Bi, Gamma);
    lhat = eps_lse * log(sum(exp(-alpha/eps_lse)));
    val = rho1*e1 + rho2*lhat;
end

function [f, G] = f2_grad(T, R, Bi, Gamma, N0, rho1, rho2, eps_lse)
    % paper (42): ∇f2 = 4rho1(WW-R)T -2rho2 * sum exp(-a/eps) Gi / sum exp(-a/eps)
    W = T*T';
    alpha = alpha_total(T, Bi, Gamma);
    lhat = eps_lse * log(sum(exp(-alpha/eps_lse)));
    f = rho1*norm(W-R,'fro')^2 + rho2*lhat;

    term1 = 4*rho1*(W-R)*T;

    alpha = alpha - min(alpha);
    w = exp(-alpha/eps_lse);
    w = w / (sum(w)+eps);

    K = size(T,2);
    term2 = zeros(size(T));
    for i=1:K
        ti = T(:,i);
        ei = zeros(K,1); ei(i)=1;
        Gi = Bi{i} * ( (1+Gamma(i))*ti*ei.' - Gamma(i)*T );
        term2 = term2 + w(i)*Gi;
    end
    G = term1 - 2*rho2*term2;
end

function alpha = alpha_total(T, Bi, Gamma)
    % paper (36): alpha_i = (1+G_i) tr(Bi ti ti^H) - G_i tr(Bi T T^H)
    K = size(T,2);
    W = T*T';
    alpha = zeros(K,1);
    for i=1:K
        ti = T(:,i);
        alpha(i) = (1+Gamma(i))*real(trace(Bi{i}*(ti*ti'))) - Gamma(i)*real(trace(Bi{i}*W));
    end
end

%% ---------------- RCG on oblique manifold (Per-antenna) ----------------
function X = rcg_oblique_sum(X0, R, H, Gamma, rho, P0, eps_lse, cfg, N0)
    % Solve (56) on oblique manifold:
    % X in C^{KxN}, diag(X^H X)=P0/N
    rho1=rho(1); rho2=rho(2);
    X = X0;

    g_prev=[]; d_prev=[];

    for it=1:cfg.maxit
        [f, G] = f3_grad(X, R, H, Gamma, rho1, rho2, N0);  % Euclidean gradient (58)
        gR = proj_oblique(X, G);                           % (61)

        if norm(gR,'fro') < cfg.tol, break; end

        if it==1
            d = -gR;
        else
            g_prev_tr = proj_oblique(X, g_prev);          % transport (64)
            beta = innerp(gR, gR - g_prev_tr) / (innerp(g_prev_tr,g_prev_tr)+eps);
            beta = max(beta,0);
            d = -gR + beta*d_prev;
        end

        step = 1;
        c1 = 1e-4;
        for ls=1:30
            Xnew = retract_oblique(X, step*d, P0);
            fnew = f3_only(Xnew, R, H, Gamma, rho1, rho2, N0);
            if fnew <= f + c1*step*innerp(gR,d), break; end
            step = step/2;
        end

        X = retract_oblique(X, step*d, P0);
        if any(isnan(X(:))) || any(isinf(X(:)))
    warning('NaN detected in Max-RCG (Per-antenna), aborting this run');
    break;
        end
        g_prev = G;
        d_prev = d;
    end
end

function X = rcg_oblique_max(X0, R, H, Gamma, rho, P0, eps_lse, cfg, N0)
    rho1=rho(1); rho2=rho(2);
    X = X0;

    g_prev=[]; d_prev=[];

    for it=1:cfg.maxit
        [f, G] = f4_grad(X, R, H, Gamma, rho1, rho2, eps_lse, N0); % (59)
        gR = proj_oblique(X, G);

        if norm(gR,'fro') < cfg.tol, break; end

        if it==1
            d = -gR;
        else
            g_prev_tr = proj_oblique(X, g_prev);
            beta = innerp(gR, gR - g_prev_tr) / (innerp(g_prev_tr,g_prev_tr)+eps);
            beta = max(beta,0);
            d = -gR + beta*d_prev;
        end

        step = 1; c1=1e-4;
        for ls=1:30
            Xnew = retract_oblique(X, step*d, P0);        % (62)
            fnew = f4_only(Xnew, R, H, Gamma, rho1, rho2, eps_lse, N0);
            if fnew <= f + c1*step*innerp(gR,d), break; end
            step = step/2;
        end

        X = retract_oblique(X, step*d, P0);
        g_prev = G;
        d_prev = d;
    end
end

function gR = proj_oblique(X, G)
    % paper (61): grad = G - X * ddiag(Re(X^H G))
    D = real(diag(X'*G));
    gR = G - X*diag(D);
end

function Xnew = retract_oblique(X, Xi, P0)
    % paper (62): normalize each column to sqrt(P0/N)
    Y = X + Xi;
    N = size(X,2);
    Xnew = Y;
    for n=1:N
        Xnew(:,n) = Y(:,n) / (norm(Y(:,n))+eps) * sqrt(P0/N);
    end
end

function val = f3_only(X, R, H, Gamma, rho1, rho2, N0)
    % f3 = rho1||X^H X - R||^2 + rho2||alpha - N0*Gamma||^2
    W = X'*X;
    a = alpha_oblique(X, H, Gamma); % (54)
    val = rho1*norm(W-R,'fro')^2 + rho2*norm(a - N0*Gamma,'fro')^2;
end

function [f, G] = f3_grad(X, R, H, Gamma, rho1, rho2, N0)
    % paper (58): ∇f3 = 4rho1 X (X^H X - R) + 4rho2 sum_i alpha_i G_i^H
    W = X'*X;
    a = alpha_oblique(X, H, Gamma);
    f = rho1*norm(W-R,'fro')^2 + rho2*norm(a - N0*Gamma,'fro')^2;

    term1 = 4*rho1 * X * (W - R);

    % Build Gi in X-form:
    % From paper: Gi defined by (37) and can be transformed as function of X.
    % We implement via equivalent T-form using T = X^T and then map back.
    T = X.'; % N x K
    Bi = cell(numel(Gamma),1);
    for i=1:numel(Gamma)
        hi = H(:,i);
        Bi{i} = conj(hi)*transpose(hi);
    end

    term2_T = zeros(size(T));
    for i=1:numel(Gamma)
        ti = T(:,i);
        ei = zeros(numel(Gamma),1); ei(i)=1;
        GiT = Bi{i} * ( (1+Gamma(i))*ti*ei.' - Gamma(i)*T );
        term2_T = term2_T + a(i)*GiT;
    end
    term2 = 4*rho2 * (term2_T.'); % back to KxN

    G = term1 + term2;
end

function val = f4_only(X, R, H, Gamma, rho1, rho2, eps_lse, N0)
    W = X'*X;
    a = alpha_oblique(X, H, Gamma);
    lhat = eps_lse * log(sum(exp(-a/eps_lse)));
    val = rho1*norm(W-R,'fro')^2 + rho2*lhat;
end

function [f, G] = f4_grad(X, R, H, Gamma, rho1, rho2, eps_lse, N0)
    % paper (59): ∇f4 = 4rho1 X(W-R) -2rho2 * sum exp(-a/eps) Gi^H / sum exp(-a/eps)
    W = X'*X;
    a = alpha_oblique(X, H, Gamma);
    lhat = eps_lse * log(sum(exp(-a/eps_lse)));
    f = rho1*norm(W-R,'fro')^2 + rho2*lhat;

    term1 = 4*rho1 * X * (W - R);

    T = X.'; % N x K
    Bi = cell(numel(Gamma),1);
    for i=1:numel(Gamma)
        hi = H(:,i);
        Bi{i} = conj(hi)*transpose(hi);
    end

    w = exp(-a/eps_lse);
    w = w/(sum(w)+eps);

    term2_T = zeros(size(T));
    for i=1:numel(Gamma)
        ti = T(:,i);
        ei = zeros(numel(Gamma),1); ei(i)=1;
        GiT = Bi{i} * ( (1+Gamma(i))*ti*ei.' - Gamma(i)*T );
        term2_T = term2_T + w(i)*GiT;
    end
    term2 = -2*rho2 * (term2_T.');  % back to KxN
    G = term1 + term2;
end

function a = alpha_oblique(X, H, Gamma)
    % paper (54):
    % alpha_i = (1+G_i) tr(Bi X^H(i,:) X(i,:)) - G_i tr(Bi X^H X)
    % where X(i,:) is i-th row of X.
    K = size(X,1);
    T = X.';         % N x K
    W = T*T';        % N x N
    a = zeros(K,1);
    for i=1:K
        hi = H(:,i);
        Bi = conj(hi)*transpose(hi);
        ti = T(:,i);
        a(i) = (1+Gamma(i))*real(trace(Bi*(ti*ti'))) - Gamma(i)*real(trace(Bi*W));
    end
end
