function fig8_full()
% ================================================================
%  Fig.8 reproduction (TWC 2018, Liu et al.)
%  "MU-MIMO Communications With MIMO Radar: From Co-Existence to Joint Transmission"
%  - Shared deployment, K = 10, N = 20
%  - Trade-off between "MSE" of beampatterns and average downlink SINR
%  - Radar-only reference beampattern: 3dB design via Problem (10)
%  - Constrained method: Problem (20) solved by SDR (CVX)
%  - Weighted methods: penalty + RCG on manifolds (self-implemented)
%
%  OUTPUT: A figure similar to paper Fig.8:
%          Y: Average MSE (dB), X: Average SINR (dB)
%          Solid: Total power constraint
%          Dashed: Per-antenna power constraint
%          Legends: Constrained, SDR; Max, RCG; Sum-Squ, RCG
%
%  REQUIREMENTS:
%    - CVX installed and in path
%
%  Notes:
%    - The paper uses many Monte-Carlo channel realizations.
%    - For the "trade-off curve", we sweep:
%        * Constrained-SDR: SINR threshold Gamma_dB list
%        * Weighted-RCG: weight ratio rho2/rho1 list
%
%  Author: ChatGPT (for Lucia), aligned to paper structure.
% ================================================================

clc; clear; close all;

%% ========== Global settings (Paper VI) ==========
P0_dBm = 20;                % total power
P0     = 10^(P0_dBm/10);    % linear
N0_dBm = 0;                 % noise power
N0     = 10^(N0_dBm/10);    % linear

N = 20;                     % antennas
K = 10;                     % users (Fig.8 uses K=10)

lambda = 1;
d = 0.5*lambda;             % half-wavelength spacing

% Angle grid (fine grid)
theta_deg = (-90:0.2:90).'; % M x 1
theta_rad = deg2rad(theta_deg);
M = numel(theta_deg);

% 3dB beampattern setup (same shape as Fig.4(b)): mainbeam at 0 deg, width 10 deg
theta0 = 0;     % center
theta1 = -5;    % -5 deg gives 10 deg 3dB width
theta2 = +5;

% Define sidelobe region Θ (exclude mainlobe interval [theta1, theta2])
SL_mask = (theta_deg <= theta1) | (theta_deg >= theta2); % simple, consistent with mainlobe exclusion

% Monte-Carlo
MC = 10;                 % increase to 200~500 for closer match (runtime increases)
rng(7);

%% ========== Build steering vectors ==========
A = steering_ula(theta_rad, N, d, lambda);     % N x M
a0 = steering_ula(deg2rad(theta0), N, d, lambda);  % N x 1
a1 = steering_ula(deg2rad(theta1), N, d, lambda);
a2 = steering_ula(deg2rad(theta2), N, d, lambda);

%% ========== Step 1: Radar-only covariance R via Problem (10) ==========
% Problem (10): maximize t subject to mainlobe-to-sidelobe gap and 3dB constraints
fprintf('Solving radar-only (10) via CVX...\n');
R = radar_cov_3db_cvx(N, P0, a0, a1, a2, A, SL_mask);

% Radar-only beampattern (reference)
P_radar = real(diag(A' * R * A));   % M x 1

%% ========== Trade-off sweep parameters ==========
% (A) Constrained-SDR curve: sweep SINR thresholds
Gamma_dB_list = 6:1:11;                    % paper Fig.8 x-range ~ 6-11 dB
Gamma_list    = 10.^(Gamma_dB_list/10);

% (B) Weighted-RCG curves: sweep weight ratio rho2/rho1
% Keep rho1 fixed and sweep rho2 to trade SINR vs beampattern fit.
%rho1 = 10;
%rho2_list_sum = [0.05 0.1 0.2 0.4 0.8 1.5 3 6 10 20];  % Sum-Squ
rho1 = 10;
rho2_list_sum = [0.05 0.1 0.2 0.4 0.8 1 1.5 2 3];

rho2_list_max = [0.05 0.1 0.2 0.4 0.8 1.5 3 6 10 20];  % Max penalty
eps_lse = 0.3; % log-sum-exp smoothing epsilon (small positive)

%% ========== Containers for results ==========
% Each curve returns a set of points (AvgSINR_dB, AvgMSE_dB)
res_con_total = zeros(numel(Gamma_list), 2);
res_con_pa    = zeros(numel(Gamma_list), 2);

res_sumsq_total = zeros(numel(rho2_list_sum), 2);
res_sumsq_pa    = zeros(numel(rho2_list_sum), 2);

res_max_total = zeros(numel(rho2_list_max), 2);
res_max_pa    = zeros(numel(rho2_list_max), 2);

%% ========== (A) Constrained, SDR ==========
fprintf('\n=== Constrained, SDR: sweeping Gamma (CVX SDP) ===\n');
for ig = 1:numel(Gamma_list)
    Gamma = Gamma_list(ig) * ones(K,1);

    sinr_mc_total = zeros(MC,1);
    mse_mc_total  = zeros(MC,1);

    sinr_mc_pa = zeros(MC,1);
    mse_mc_pa  = zeros(MC,1);

    for mc = 1:MC
        H = (randn(N,K)+1j*randn(N,K))/sqrt(2); % i.i.d. CN(0,1)

        % ---- Total-power constrained variant (relax diag -> trace) ----
        [T_total] = solve_constrained_sdr_total(H, R, P0, N0, Gamma);
        [sinr_mc_total(mc), mse_mc_total(mc)] = metrics_sinr_mse(H, T_total, N0, A, P_radar);

        % ---- Per-antenna constrained (diag equality) ----
        [T_pa] = solve_constrained_sdr_perant(H, R, P0, N0, Gamma);
        [sinr_mc_pa(mc), mse_mc_pa(mc)] = metrics_sinr_mse(H, T_pa, N0, A, P_radar);
    end

    res_con_total(ig,1) = 10*log10(mean(sinr_mc_total));
    res_con_total(ig,2) = 10*log10(mean(mse_mc_total));

    res_con_pa(ig,1) = 10*log10(mean(sinr_mc_pa));
    res_con_pa(ig,2) = 10*log10(mean(mse_mc_pa));

    fprintf('Gamma=%.1f dB -> Total:(%.2f, %.2f), PerAnt:(%.2f, %.2f)\n',...
        Gamma_dB_list(ig), res_con_total(ig,1), res_con_total(ig,2), res_con_pa(ig,1), res_con_pa(ig,2));
end

%% ========== (B) Weighted, RCG (Sum-Squ) ==========
fprintf('\n=== Weighted, RCG: Sum-Squ penalty sweeping rho2 ===\n');
for ir = 1:numel(rho2_list_sum)
    rho = [rho1, rho2_list_sum(ir)];

    sinr_mc_total = zeros(MC,1);
    mse_mc_total  = zeros(MC,1);

    sinr_mc_pa = zeros(MC,1);
    mse_mc_pa  = zeros(MC,1);

    for mc = 1:MC
        H = (randn(N,K)+1j*randn(N,K))/sqrt(2);

        % total power manifold (complex sphere)
        T_total = solve_weighted_rcg_total(H, R, P0, N0, Gamma_list(1)*ones(K,1), rho, "sumsq", eps_lse);
        [sinr_mc_total(mc), mse_mc_total(mc)] = metrics_sinr_mse(H, T_total, N0, A, P_radar);

        % per-antenna manifold (row norms fixed)
        T_pa = solve_weighted_rcg_perant(H, R, P0, N0, Gamma_list(1)*ones(K,1), rho, "sumsq", eps_lse);
        [sinr_mc_pa(mc), mse_mc_pa(mc)] = metrics_sinr_mse(H, T_pa, N0, A, P_radar);
    end

    res_sumsq_total(ir,1) = 10*log10(mean(sinr_mc_total));
    res_sumsq_total(ir,2) = 10*log10(mean(mse_mc_total));

    res_sumsq_pa(ir,1) = 10*log10(mean(sinr_mc_pa));
    res_sumsq_pa(ir,2) = 10*log10(mean(mse_mc_pa));

    fprintf('rho2=%.3g -> Total:(%.2f, %.2f), PerAnt:(%.2f, %.2f)\n',...
        rho2_list_sum(ir), res_sumsq_total(ir,1), res_sumsq_total(ir,2), res_sumsq_pa(ir,1), res_sumsq_pa(ir,2));
end

%% ========== (C) Weighted, RCG (Max penalty) ==========
fprintf('\n=== Weighted, RCG: Max penalty sweeping rho2 ===\n');
for ir = 1:numel(rho2_list_max)
    rho = [rho1, rho2_list_max(ir)];

    sinr_mc_total = zeros(MC,1);
    mse_mc_total  = zeros(MC,1);

    sinr_mc_pa = zeros(MC,1);
    mse_mc_pa  = zeros(MC,1);

    for mc = 1:MC
        H = (randn(N,K)+1j*randn(N,K))/sqrt(2);

        T_total = solve_weighted_rcg_total(H, R, P0, N0, Gamma_list(1)*ones(K,1), rho, "max", eps_lse);
        [sinr_mc_total(mc), mse_mc_total(mc)] = metrics_sinr_mse(H, T_total, N0, A, P_radar);

        T_pa = solve_weighted_rcg_perant(H, R, P0, N0, Gamma_list(1)*ones(K,1), rho, "max", eps_lse);
        [sinr_mc_pa(mc), mse_mc_pa(mc)] = metrics_sinr_mse(H, T_pa, N0, A, P_radar);
    end

    res_max_total(ir,1) = 10*log10(mean(sinr_mc_total));
    res_max_total(ir,2) = 10*log10(mean(mse_mc_total));

    res_max_pa(ir,1) = 10*log10(mean(sinr_mc_pa));
    res_max_pa(ir,2) = 10*log10(mean(mse_mc_pa));

    fprintf('rho2=%.3g -> Total:(%.2f, %.2f), PerAnt:(%.2f, %.2f)\n',...
        rho2_list_max(ir), res_max_total(ir,1), res_max_total(ir,2), res_max_pa(ir,1), res_max_pa(ir,2));
end

%% ========== Plot (match Fig.8 style) ==========
figure; hold on; grid on;

% Sort points by SINR to draw smooth-looking curves
plot_sorted(res_max_total,  '-x',  'Max, RCG (Total)');
plot_sorted(res_sumsq_total,'-o',  'Sum-Squ, RCG (Total)');
plot_sorted(res_con_total,  '-^',  'Constrained, SDR (Total)');

plot_sorted(res_max_pa,     '--x', 'Max, RCG (Per-Ant)');
plot_sorted(res_sumsq_pa,   '--o', 'Sum-Squ, RCG (Per-Ant)');
plot_sorted(res_con_pa,     '--^', 'Constrained, SDR (Per-Ant)');

xlabel('Average SINR (dB)');
ylabel('Average MSE (dB)');
title('Fig.8 Reproduction: MSE vs SINR (Shared Deployment, K=10)');
legend('Location','best');
xlim([5.5 12]);  % paper-like range
% ylim is data-dependent; let MATLAB decide

end

%% ======================= Helper: steering vector =======================
function A = steering_ula(theta_rad, N, d, lambda)
% theta_rad can be scalar or vector
u = sin(theta_rad(:)).';     % 1 x M
n = (0:N-1).';               % N x 1
A = exp(1j*2*pi*(d/lambda) * n * u); % N x M
end

%% ======================= Radar-only (10) via CVX =======================
function R = radar_cov_3db_cvx(N, P0, a0, a1, a2, A, SL_mask)
% Solve Stoica-type 3dB design:
% max t  s.t.  a0^H R a0 - a(theta)^H R a(theta) >= t  for theta in sidelobe region
%             a1^H R a1 = a0^H R a0 / 2
%             a2^H R a2 = a0^H R a0 / 2
%             R >=0, Hermitian, diag(R)=P0/N
%
% Implementation in CVX: maximize t (equiv min -t)

M = size(A,2);
idx = find(SL_mask);

cvx_begin sdp quiet
    variable R(N,N) hermitian semidefinite
    variable t
    % Equal per-antenna power
    diag(R) == (P0/N)*ones(N,1);

    % 3dB constraints
    real(a1' * R * a1) == real(a0' * R * a0)/2;
    real(a2' * R * a2) == real(a0' * R * a0)/2;

    % Sidelobe gap constraints
    for ii = 1:numel(idx)
        m = idx(ii);
        real(a0' * R * a0) - real(A(:,m)' * R * A(:,m)) >= t;
    end

    maximize(t)
cvx_end

end

%% ======================= Constrained SDR (Total) =======================
function T = solve_constrained_sdr_total(H, R, P0, N0, Gamma)
% Solve relaxed SDP version of (20) under total power constraint:
%   min ||C - R||_F^2
%   s.t. SINR_i >= Gamma_i
%        tr(C) = P0
%        Tk >=0, C = sum Tk
%
% Return a rank-1 approximate beamformer matrix T (N x K).

[N,K] = size(H);

% Precompute Bi = h_i^* h_i^T
B = cell(K,1);
for i = 1:K
    hi = H(:,i);
    B{i} = conj(hi) * (hi.'); % N x N
end

cvx_begin sdp quiet
    variable C(N,N) hermitian semidefinite
    variable Tk(N,N,K) hermitian semidefinite

    % objective ||C-R||_F^2
    minimize( square_pos(norm(C - R,'fro')) )

    % coupling
    C == sum(Tk,3);

    % total power equality
    trace(C) == P0;

    % SINR constraints using covariance form:
    % tr(Bi*Ti) >= Gamma_i*( tr(Bi*(C - Ti)) + N0 )
    for i = 1:K
        Ti = Tk(:,:,i);
        trace(B{i}*Ti) >= Gamma(i) * ( trace(B{i}*(C - Ti)) + N0 );
    end
cvx_end

% rank-1 extraction for each user
T = zeros(N,K);
for k = 1:K
    [V,D] = eig(full(Tk(:,:,k)));
    [dmax,idx] = max(real(diag(D)));
    vk = V(:,idx);
    T(:,k) = sqrt(max(dmax,0)) * vk;
end

% normalize total power to exactly P0
T = T * sqrt(P0 / (norm(T,'fro')^2 + 1e-12));
end

%% =================== Constrained SDR (Per-Antenna) =====================
function T = solve_constrained_sdr_perant(H, R, P0, N0, Gamma)
% Solve relaxed SDP version of (20) under per-antenna equality:
%   diag(C) = P0/N * 1
% Return rank-1 approximate T (N x K), then row-normalize to satisfy per-antenna power.

[N,K] = size(H);

B = cell(K,1);
for i = 1:K
    hi = H(:,i);
    B{i} = conj(hi) * (hi.'); % N x N
end

cvx_begin sdp quiet
    variable C(N,N) hermitian semidefinite
    variable Tk(N,N,K) hermitian semidefinite

    minimize( square_pos(norm(C - R,'fro')) )

    C == sum(Tk,3);

    % per-antenna power equality
    diag(C) == (P0/N)*ones(N,1);

    for i = 1:K
        Ti = Tk(:,:,i);
        trace(B{i}*Ti) >= Gamma(i) * ( trace(B{i}*(C - Ti)) + N0 );
    end
cvx_end

T = zeros(N,K);
for k = 1:K
    [V,D] = eig(full(Tk(:,:,k)));
    [dmax,idx] = max(real(diag(D)));
    vk = V(:,idx);
    T(:,k) = sqrt(max(dmax,0)) * vk;
end

% enforce per-antenna equal power by row normalization
rowPow = sum(abs(T).^2,2);                   % N x 1
target = (P0/N);
scale  = sqrt(target ./ (rowPow + 1e-12));   % N x 1
T = diag(scale) * T;

end

%% ===================== Metrics: AvgSINR & "MSE" ========================
function [avgSINR, mse_val] = metrics_sinr_mse(H, T, N0, A, P_radar)
% avgSINR: average over users (linear), with SINR defined from beamformers
% mse_val: sum_m (P_radar(theta_m) - P_radcom(theta_m))^2

[N,K] = size(H);

% SINR per user
sinr_i = zeros(K,1);
for i = 1:K
    hi = H(:,i);
    num = abs(hi.' * T(:,i))^2;
    den = N0;
    for k = 1:K
        if k~=i
            den = den + abs(hi.' * T(:,k))^2;
        end
    end
    sinr_i(i) = num / max(den,1e-12);
end
avgSINR = mean(sinr_i);

% Beampattern of RadCom
C = T*T';
P_radcom = real(diag(A' * C * A));

% Paper's "MSE" definition (sum of squared error over grid)
mse_val = sum( (P_radar - P_radcom).^2 );

end

%% ==================== Weighted RCG (Total power sphere) =================
function T = solve_weighted_rcg_total(H, R, P0, N0, Gamma, rho, mode, eps_lse)
% Solve weighted problem on complex sphere:
%   min rho1||TT^H - R||^2 + rho2 * penalty(alpha)
% s.t. ||T||_F^2 = P0
%
% mode: "sumsq" or "max"

[N,K] = size(H);
rho1 = rho(1); rho2 = rho(2);

% init random on sphere
T = (randn(N,K)+1j*randn(N,K))/sqrt(2);
T = T * sqrt(P0/(norm(T,'fro')^2 + 1e-12));

maxit = 120;
tol = 1e-6;

% initial direction
[egrad] = egrad_weighted(H,T,R,N0,Gamma,rho1,rho2,mode,eps_lse);
rgrad = proj_sphere(T, egrad);
D = -rgrad;

for it = 1:maxit
    ng = norm(rgrad,'fro');
    if ng < tol, break; end

    f0 = cost_weighted(H,T,R,N0,Gamma,rho1,rho2,mode,eps_lse);

    % Armijo backtracking
    step = 1;
    c1 = 1e-4; tau = 0.5;
    inner = real(trace(rgrad' * D));
    if inner > 0, D = -rgrad; inner = -ng^2; end

    while true
        Tnew = retract_sphere(T, step, D, P0);
        fnew = cost_weighted(H,Tnew,R,N0,Gamma,rho1,rho2,mode,eps_lse);
        if fnew <= f0 + c1*step*inner || step < 1e-8
            break;
        end
        step = step * tau;
    end

    % new gradient
    egrad_new = egrad_weighted(H,Tnew,R,N0,Gamma,rho1,rho2,mode,eps_lse);
    rgrad_new = proj_sphere(Tnew, egrad_new);

    % vector transport (project old D to new tangent)
    D_trans = proj_sphere(Tnew, D);

    % Polak-Ribiere (Riemannian)
    y = rgrad_new - proj_sphere(Tnew, rgrad); % transported rgrad
    beta = real(trace(rgrad_new' * y)) / (real(trace(rgrad' * rgrad)) + 1e-12);
    beta = max(beta, 0);

    D = -rgrad_new + beta * D_trans;

    T = Tnew;
    rgrad = rgrad_new;
end

end

%% ================ Weighted RCG (Per-antenna row-norm manifold) ==========
function T = solve_weighted_rcg_perant(H, R, P0, N0, Gamma, rho, mode, eps_lse)
% Per-antenna equal power: each antenna (row) has power P0/N:
%   sum_k |T(n,k)|^2 = P0/N
% We run RCG with row-wise projection & retraction.

[N,K] = size(H);
rho1 = rho(1); rho2 = rho(2);
row_target = P0/N;

% init random with row normalization
T = (randn(N,K)+1j*randn(N,K))/sqrt(2);
T = retract_row_norm(T, row_target);

maxit = 140;
tol = 1e-6;

egrad = egrad_weighted(H,T,R,N0,Gamma,rho1,rho2,mode,eps_lse);
rgrad = proj_row_manifold(T, egrad, row_target);
D = -rgrad;

for it = 1:maxit
    ng = norm(rgrad,'fro');
    if ng < tol, break; end

    f0 = cost_weighted(H,T,R,N0,Gamma,rho1,rho2,mode,eps_lse);

    step = 1;
    c1 = 1e-4; tau = 0.5;
    inner = real(trace(rgrad' * D));
    if inner > 0, D = -rgrad; inner = -ng^2; end

    while true
        Tnew = retract_row_norm(T + step*D, row_target);
        fnew = cost_weighted(H,Tnew,R,N0,Gamma,rho1,rho2,mode,eps_lse);
        if fnew <= f0 + c1*step*inner || step < 1e-8
            break;
        end
        step = step * tau;
    end

    egrad_new = egrad_weighted(H,Tnew,R,N0,Gamma,rho1,rho2,mode,eps_lse);
    rgrad_new = proj_row_manifold(Tnew, egrad_new, row_target);

    D_trans = proj_row_manifold(Tnew, D, row_target);

    y = rgrad_new - proj_row_manifold(Tnew, rgrad, row_target);
    beta = real(trace(rgrad_new' * y)) / (real(trace(rgrad' * rgrad)) + 1e-12);
    beta = max(beta, 0);

    D = -rgrad_new + beta * D_trans;

    T = Tnew;
    rgrad = rgrad_new;
end

end

%% ======================== Cost & Gradient (Euclidean) ===================
function f = cost_weighted(H,T,R,N0,Gamma,rho1,rho2,mode,eps_lse)
C = T*T';
f_radar = rho1 * norm(C - R,'fro')^2;

alpha = alpha_vec(H,T,Gamma); % Kx1
if strcmpi(mode,"sumsq")
    pen = sum( (alpha - Gamma*N0).^2 );
elseif strcmpi(mode,"max")
    % smooth max of (-alpha)
    z = -alpha / max(eps_lse,1e-6);
    pen = eps_lse * log(sum(exp(z)));
else
    error('Unknown mode');
end
f = f_radar + rho2*pen;
end

function egrad = egrad_weighted(H,T,R,N0,Gamma,rho1,rho2,mode,eps_lse)
% Implements paper-like gradients:
% radar part: 4*rho1*(TT^H - R)*T
% penalty part uses Gi and alpha_i weights

[N,K] = size(T);
C = T*T';
egrad_radar = 4*rho1*(C - R)*T;

alpha = alpha_vec(H,T,Gamma); % Kx1

% Build Bi and Gi efficiently:
% Bi = conj(h_i)*h_i.' (N x N)
% Gi = Bi * ((1+Gamma_i)*t_i*e_i^T - Gamma_i*T)
Gsum = zeros(N,K);

if strcmpi(mode,"sumsq")
    w = 4*rho2*(alpha - Gamma*N0);  % Kx1 coefficient for each Gi

    for i = 1:K
        hi = H(:,i);
        Bi = conj(hi) * (hi.');
        Ei = zeros(1,K); Ei(i)=1;
        Gi = Bi * ( (1+Gamma(i))* (T(:,i)*Ei) - Gamma(i)*T );
        Gsum = Gsum + w(i)*Gi;
    end
    egrad = egrad_radar + Gsum;

elseif strcmpi(mode,"max")
    % smooth max: weights proportional to exp(-alpha/eps)
    z = -alpha / max(eps_lse,1e-6);
    s = exp(z);
    s = s / (sum(s) + 1e-12);  % softmax weights

    for i = 1:K
        hi = H(:,i);
        Bi = conj(hi) * (hi.');
        Ei = zeros(1,K); Ei(i)=1;
        Gi = Bi * ( (1+Gamma(i))* (T(:,i)*Ei) - Gamma(i)*T );
        Gsum = Gsum + s(i)*Gi;
    end
    % paper (42) has "-2*rho2 * weighted_sum(Gi)"
    egrad = egrad_radar - 2*rho2*Gsum;

else
    error('Unknown mode');
end

end

function alpha = alpha_vec(H,T,Gamma)
% alpha_i = (1+Gamma_i)*|h_i^T t_i|^2 - Gamma_i * sum_k |h_i^T t_k|^2
[N,K] = size(H);
alpha = zeros(K,1);
for i = 1:K
    hi = H(:,i);
    s_all = hi.' * T;           % 1 x K, complex
    p_all = abs(s_all).^2;      % 1 x K
    alpha(i) = (1+Gamma(i))*p_all(i) - Gamma(i)*sum(p_all);
end
end

%% ======================= Manifold ops: Sphere ==========================
function rgrad = proj_sphere(T, egrad)
% Projection onto tangent: egrad - Re(tr(T^H egrad))*T
c = real(trace(T' * egrad));
rgrad = egrad - c*T;
end

function Tnew = retract_sphere(T, step, D, P0)
Y = T + step*D;
Tnew = Y * sqrt(P0/(norm(Y,'fro')^2 + 1e-12));
end

%% =================== Manifold ops: Row-norm (Per-Ant) ===================
function rgrad = proj_row_manifold(T, egrad, row_target)
% Tangent condition per row n: Re( t_n * f_n^H ) = 0
% Projection row-wise:
[N,~] = size(T);
rgrad = zeros(size(T));
for n = 1:N
    tn = T(n,:);
    fn = egrad(n,:);
    denom = row_target; % ||tn||^2 ideally equals row_target
    coeff = real(fn * tn') / (denom + 1e-12);
    rgrad(n,:) = fn - coeff*tn;
end
end

function Tn = retract_row_norm(T, row_target)
% normalize each row to have squared norm row_target
rowPow = sum(abs(T).^2,2);
scale = sqrt(row_target ./ (rowPow + 1e-12));
Tn = diag(scale) * T;
end

%% ======================= Plot helper =======================
function plot_sorted(XY, style, nameStr)
% XY: [x y]
[~,idx] = sort(XY(:,1));
plot(XY(idx,1), XY(idx,2), style, 'LineWidth', 1.3, 'DisplayName', nameStr);
end
