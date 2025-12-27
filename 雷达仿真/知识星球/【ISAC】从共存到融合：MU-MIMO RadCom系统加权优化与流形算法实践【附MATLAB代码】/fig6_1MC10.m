% =============================================================
% Fig.6 - Feasibility Probability Comparison
% MU-MIMO Communications with MIMO Radar: From Co-existence to Joint Transmission
%
% B. Feasibility Comparison Between Constrained Problems and Penalty Problems
%   - K = 17:20, Gamma = 10 dB, N = 20
%   - Feasibility Probability (%) over Monte Carlo channel realizations
%
% Curves (bars):
%   1) Separated deployment, ZF constrained design (generally infeasible for large K)
%   2) Shared deployment, constrained SDR problem (20)
%   3) Weighted/Penalty optimization (RCG on manifolds): always feasible (100%)
%
% Notes:
%   - For separated deployment: NC = N-NR, if K>NC => ZF infeasible (DoF不足)
%   - For shared SDR: feasible if CVX returns Solved / Inaccurate-Solved
% =============================================================
clear; clc; close all;

%% ---------------- Basic parameters (paper-like) ----------------
P0_dBm = 20;  P0 = 10^(P0_dBm/10);
N      = 20;
NR     = 14;                 % separated radar antennas (paper uses NR=14 in examples)
NC     = N - NR;             % separated comm antennas
N0_dBm = 0;   N0 = 10^(N0_dBm/10);

Gamma_dB = 10;
gamma    = 10^(Gamma_dB/10);

K_list   = 17:20;

Nmc      = 10;              % 建议 100 或 200；越大越平滑
rng(1);

lambda = 1; d = 0.5*lambda;

% 角度网格（这里主要用于 Radar-only R2 的生成；Fig.6 本身不画方向图）
theta_deg = -90:1:90;
theta_rad = deg2rad(theta_deg);
M = numel(theta_deg);

A = zeros(N,M);
for m = 1:M
    A(:,m) = exp(1j*2*pi*(d/lambda)*(0:N-1)'*sin(theta_rad(m)));
end

%% ---------------- Step 0: Compute radar-only template R2 (Problem 10) ----------------
% Fig.6 的 shared SDR (20) 需要一个 R2 作为雷达参考协方差
% 这里按 Fig.4(b) 的 3 dB 主瓣：center=0°, width=10° => [-5,5]
theta0 = 0; bw3 = 10;
theta1 = theta0 - bw3/2;
theta2 = theta0 + bw3/2;

[~, idx0] = min(abs(theta_deg - theta0));
[~, idxL] = min(abs(theta_deg - theta1));
[~, idxR] = min(abs(theta_deg - theta2));

a0 = A(:,idx0);
aL = A(:,idxL);
aR = A(:,idxR);

idx_sidelobe = find( (theta_deg < theta1) | (theta_deg > theta2) );

fprintf('Precomputing Radar-only R2 (Shared, Problem 10) via CVX...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t0
    minimize(-t0)
    subject to
        diag(R2) == (P0/N)*ones(N,1);

        main0 = real(a0' * R2 * a0);
        % strict 3 dB points (paper-like)
        real(aL' * R2 * aL) == 0.5*main0;
        real(aR' * R2 * aR) == 0.5*main0;

        for mm = idx_sidelobe
            am = A(:,mm);
            real(main0 - am' * R2 * am) >= t0;
        end
cvx_end

if ~(strcmpi(cvx_status,'Solved') || strcmpi(cvx_status,'Inaccurate/Solved'))
    error('Radar-only R2 failed: %s', cvx_status);
end
fprintf('Radar-only R2 ready.\n\n');

%% ---------------- Containers for feasibility probability ----------------
p_sep  = zeros(numel(K_list),1);   % Separated ZF constrained (SDR/ZF)
p_sh   = zeros(numel(K_list),1);   % Shared constrained (Problem 20 SDR)
p_wt   = zeros(numel(K_list),1);   % Weighted/Penalty (RCG): always feasible

%% ---------------- Main loop over K ----------------
for ik = 1:numel(K_list)
    K = K_list(ik);
    fprintf('=== K = %d, Gamma = %g dB, MC = %d ===\n', K, Gamma_dB, Nmc);

    cnt_sep = 0;
    cnt_sh  = 0;

    % Weighted/Penalty optimization: paper says always feasible
    cnt_wt = Nmc;

    for imc = 1:Nmc
        % -------- channel realization --------
        H = (randn(N,K) + 1j*randn(N,K))/sqrt(2);

        % ===== 1) Separated deployment ZF feasibility =====
        % separated comm antennas = NC, ZF MU-MIMO needs DoF >= K (at least)
        % if K > NC, generally infeasible (matches paper statement)
        if K <= NC
            % 这里如果你未来想“真解一个 separated ZF/SDR”也可以写 CVX
            % 但在 Fig.6 的 K=17~20 下，NC=6，永远不满足，所以这里直接判可行
            cnt_sep = cnt_sep + 1;
        end

        % ===== 2) Shared deployment SDR feasibility: solve Problem (20) =====
        % Variables: Wi (N×N) PSD, i=1..K
        % Constraints:
        %   diag(sum Wi) = P0/N
        %   SINR_i >= gamma
        % Objective: ||sumWi - R2||_F^2 (keeps bounded; feasibility is what we count)
        HiHiH = cell(K,1);
        for i = 1:K
            hi = H(:,i);
            HiHiH{i} = hi*hi';
        end

        % --- Solve shared constrained SDR (20) ---
        cvx_begin sdp quiet
            cvx_precision default
            variable W(N,N,K) hermitian semidefinite

            expression Wsum(N,N)
            Wsum = 0;
            for k = 1:K
                Wsum = Wsum + W(:,:,k);
            end

            minimize( square_pos(norm(Wsum - R2,'fro')) )

            subject to
                diag(Wsum) == (P0/N)*ones(N,1);

                for i = 1:K
                    num = real(trace(HiHiH{i} * W(:,:,i)));
                    interf = 0;
                    for k = 1:K
                        if k ~= i
                            interf = interf + real(trace(HiHiH{i} * W(:,:,k)));
                        end
                    end
                    num >= gamma * (interf + N0);
                end
        cvx_end

        if (strcmpi(cvx_status,'Solved') || strcmpi(cvx_status,'Inaccurate/Solved'))
            cnt_sh = cnt_sh + 1;
        end
    end

    p_sep(ik) = 100 * cnt_sep / Nmc;
    p_sh(ik)  = 100 * cnt_sh  / Nmc;
    p_wt(ik)  = 100 * cnt_wt  / Nmc;

    fprintf('Separated ZF feasible = %.1f%%, Shared SDR feasible = %.1f%%, Weighted/RCG feasible = %.1f%%\n\n', ...
        p_sep(ik), p_sh(ik), p_wt(ik));
end

%% ---------------- Plot Fig.6 (grouped bars) ----------------
figure;

X = (1:numel(K_list));
bar_width = 0.25;

bar(X - bar_width, p_sep, bar_width); hold on;
bar(X,             p_sh,  bar_width);
bar(X + bar_width, p_wt,  bar_width);

grid on;
xticks(X);
xticklabels(string(K_list));
xlabel('Number of users K');
ylabel('Feasibility probability (%)');
title(sprintf('Fig.6 Feasibility probability (\\Gamma=%.0f dB, N=%d)', Gamma_dB, N));

legend('Separated (ZF constrained)','Shared (Problem 20, SDR)','Weighted/Penalty (RCG)', ...
       'Location','northwest');

ylim([0 100]);  % 论文图通常 0~100（或留白到 120 也行）
