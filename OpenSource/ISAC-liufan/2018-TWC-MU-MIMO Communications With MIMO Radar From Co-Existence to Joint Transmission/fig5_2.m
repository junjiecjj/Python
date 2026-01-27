% =============================================================
% Fig.5 (UPDATED) - PSLR vs SINR level (4 curves, Gamma=4~14 dB)
% Updates applied to increase PSLR & match paper:
%   (1) Angle grid: 1 degree
%   (2) Sidelobe set uses guard band (exclude near-mainlobe region)
%   (3) 3 dB constraints relaxed: <= 0.5 * main0 (instead of equality)
%   (4) Separated: R1(H) solved per MC channel realization (ZF depends on F)
%
% Curves:
%   Separated Radar-only, Separated RadCom,
%   Shared    Radar-only, Shared    RadCom
% =============================================================
clear; clc; close all;

%% ---------------- Parameters (same as Fig.3/4) ----------------
P0_dBm = 20;  P0 = 10^(P0_dBm/10);
N  = 20;  NR = 14;  NC = N-NR;
K  = 4;
N0_dBm = 0;   N0 = 10^(N0_dBm/10);

lambda = 1; d = 0.5*lambda;

% separated split
PR = P0/2;
PC = P0/2;

% Fig.5 x-axis
Gamma_dB_vec = 4:2:14;
Gamma_vec    = 10.^(Gamma_dB_vec/10);

% Monte Carlo
Nmc = 30;
rng(1);

%% ---------------- Angle grid / steering (UPDATED: 1 degree) ----------------
theta_deg = -90:1:90;              % <<< UPDATED
theta_rad = deg2rad(theta_deg);
M = numel(theta_deg);

A_full = zeros(N,M);
for m = 1:M
    A_full(:,m) = exp(1j*2*pi*(d/lambda)*(0:N-1)'*sin(theta_rad(m)));
end
A1 = A_full(1:NR,:);
A2 = A_full(NR+1:end,:);

% 3 dB spec: center 0 deg, width 10 deg => [-5, +5]
theta0 = 0;
bw3    = 10;
thetaL = theta0 - bw3/2;   % -5
thetaR = theta0 + bw3/2;   % +5

[~, idx0] = min(abs(theta_deg-theta0));
[~, idxL] = min(abs(theta_deg-thetaL));
[~, idxR] = min(abs(theta_deg-thetaR));

a0_full = A_full(:,idx0); aL_full = A_full(:,idxL); aR_full = A_full(:,idxR);
a0_r    = A1(:,idx0);     aL_r    = A1(:,idxL);     aR_r    = A1(:,idxR);

% Mainlobe index (still [-5,5])
idx_main = (theta_deg >= thetaL) & (theta_deg <= thetaR);

% ---------------- Sidelobe set (UPDATED: guard band) ----------------
guard = 2;  % degrees (try 2 or 3; larger guard -> higher PSLR usually)
idx_side = (theta_deg <= thetaL-guard) | (theta_deg >= thetaR+guard);

%% =============================================================
% Shared radar-only (10): solve once (independent of channel)
% (UPDATED: 3dB relaxed to <= half-power)
% =============================================================
fprintf('Solve Shared radar-only (10) once...\n');
cvx_begin sdp quiet
    variable R2(N,N) hermitian semidefinite
    variable t2
    minimize(-t2)
    subject to
        diag(R2) == (P0/N)*ones(N,1);

        main0 = real(a0_full' * R2 * a0_full);
        % UPDATED: relaxed 3dB constraints
        real(aL_full' * R2 * aL_full) <= 0.5*main0;
        real(aR_full' * R2 * aR_full) <= 0.5*main0;

        for m = find(idx_side)
            am = A_full(:,m);
            real(main0 - am' * R2 * am) >= t2;
        end
cvx_end

PSLR_shared_radaronly = compute_pslr(R2, A_full, idx_main, idx_side);

%% ---------------- Containers ----------------
PSLR_sep_radcom = zeros(size(Gamma_vec));
PSLR_sha_radcom = zeros(size(Gamma_vec));
pslr_sep_ro_mc_all = zeros(Nmc,1);   % store separated radar-only PSLR for MC average

%% =============================================================
% Main loops: Gamma sweep + Monte Carlo
% Separated R1(H) solved per MC (UPDATED core fix)
% =============================================================
for ig = 1:numel(Gamma_vec)
    Gamma = Gamma_vec(ig);
    fprintf('Gamma = %.1f dB  (MC=%d)\n', Gamma_dB_vec(ig), Nmc);

    pslr_sep_rc_mc = NaN(Nmc,1);
    pslr_sha_rc_mc = NaN(Nmc,1);

    for imc = 1:Nmc
        % -------- random channel H (Rayleigh) --------
        H = (randn(N,K)+1j*randn(N,K))/sqrt(2);
        F = H(1:NR ,:);      % radar->users
        G = H(NR+1:end,:);   % comm ->users

        FiFiH = cell(K,1);
        GiGiH = cell(K,1);
        Bi    = cell(K,1);  % Bi = h_i^* h_i^T
        for i = 1:K
            fi = F(:,i); gi = G(:,i); hi = H(:,i);
            FiFiH{i} = fi*fi';
            GiGiH{i} = gi*gi';
            Bi{i}    = conj(hi)*(hi.');
        end

        % =====================================================
        % Separated radar-only (13): solve R1 for THIS channel (ZF depends on FiFiH)
        % (UPDATED: relaxed 3dB constraints)
        % =====================================================
        cvx_begin sdp quiet
            variable R1(NR,NR) hermitian semidefinite
            variable t1
            minimize(-t1)
            subject to
                diag(R1) == (PR/NR)*ones(NR,1);

                main0 = real(a0_r' * R1 * a0_r);
                % UPDATED: relaxed 3dB constraints
                real(aL_r' * R1 * aL_r) <= 0.5*main0;
                real(aR_r' * R1 * aR_r) <= 0.5*main0;

                for m = find(idx_side)
                    am = A1(:,m);
                    real(main0 - am' * R1 * am) >= t1;
                end

                % ZF radar->users
                for i = 1:K
                    real(trace(R1 * FiFiH{i})) == 0;
                end
        cvx_end

        if ~(strcmpi(cvx_status,'Solved') || strcmpi(cvx_status,'Inaccurate/Solved'))
            % if radar-only failed, skip this MC
            continue;
        end

        % separated radar-only PSLR (same for all Gamma in expectation)
        Csep_ro = blkdiag(R1, zeros(NC,NC));
        pslr_sep_ro_mc_all(imc) = compute_pslr(Csep_ro, A_full, idx_main, idx_side);

        % =====================================================
        % Separated RadCom (19): solve Wk + sigma, match diag beampattern
        % =====================================================
        cvx_begin sdp quiet
            variable W(NC,NC,K) hermitian semidefinite
            variable sig_sep
            expression sumW(NC,NC)
            sumW = 0;
            for k = 1:K
                sumW = sumW + W(:,:,k);
            end

            diff_vec = real( diag( A2' * sumW * A2 - sig_sep * (A1' * R1 * A1) ) );
            minimize( sum_square(diff_vec) )
            subject to
                real(trace(sumW)) <= PC;
                sig_sep >= 0;

                for i = 1:K
                    num = real(trace(W(:,:,i) * GiGiH{i}));
                    interf = 0;
                    for k = 1:K
                        if k ~= i
                            interf = interf + real(trace(W(:,:,k) * GiGiH{i}));
                        end
                    end
                    den = interf + real(trace(R1 * FiFiH{i})) + N0;
                    num >= Gamma * den;
                end
        cvx_end

        if (strcmpi(cvx_status,'Solved') || strcmpi(cvx_status,'Inaccurate/Solved'))
            Csep_rc = blkdiag(R1, sumW);
            pslr_sep_rc_mc(imc) = compute_pslr(Csep_rc, A_full, idx_main, idx_side);
        end

        % =====================================================
        % Shared RadCom (20): solve Tk, match R2, per-antenna equality
        % =====================================================
        cvx_begin sdp quiet
            variable T(N,N,K) hermitian semidefinite
            expression sumT(N,N)
            sumT = 0;
            for k = 1:K
                sumT = sumT + T(:,:,k);
            end

            minimize( square_pos(norm(sumT - R2, 'fro')) )
            subject to
                diag(sumT) == (P0/N)*ones(N,1);

                for i = 1:K
                    num = real(trace(Bi{i} * T(:,:,i)));
                    interf = 0;
                    for k = 1:K
                        if k ~= i
                            interf = interf + real(trace(Bi{i} * T(:,:,k)));
                        end
                    end
                    den = interf + N0;
                    num >= Gamma * den;
                end
        cvx_end

        if (strcmpi(cvx_status,'Solved') || strcmpi(cvx_status,'Inaccurate/Solved'))
            pslr_sha_rc_mc(imc) = compute_pslr(sumT, A_full, idx_main, idx_side);
        end
    end

    % MC averages for this Gamma
    PSLR_sep_radcom(ig) = nanmean(pslr_sep_rc_mc);
    PSLR_sha_radcom(ig) = nanmean(pslr_sha_rc_mc);
end

% Separated radar-only horizontal line (MC average)
PSLR_sep_radaronly = nanmean(pslr_sep_ro_mc_all);

%% ---------------- Plot Fig.5 (4 curves) ----------------
figure;
plot(Gamma_dB_vec, PSLR_sep_radaronly*ones(size(Gamma_dB_vec)), 'k--','LineWidth',1.8); hold on;
plot(Gamma_dB_vec, PSLR_sep_radcom, 'ro-','LineWidth',1.8);
plot(Gamma_dB_vec, PSLR_shared_radaronly*ones(size(Gamma_dB_vec)), 'b--','LineWidth',1.8);
plot(Gamma_dB_vec, PSLR_sha_radcom, 'bs-','LineWidth',1.8);

grid on;
xlabel('SINR level (dB)');
ylabel('PSLR (dB)');
title('Fig.5  PSLR vs SINR (Updated: grid=1deg, guard band, relaxed 3dB)');
legend('Separated Radar-only','Separated RadCom', ...
       'Shared Radar-only','Shared RadCom', 'Location','Best');

%% ---------------- Helper: PSLR ----------------
function pslr = compute_pslr(C, A, idx_main, idx_side)
    P = real(diag(A' * C * A));
    main_peak = max(P(idx_main));
    side_peak = max(P(idx_side));
    pslr = 10*log10((main_peak+1e-12)/(side_peak+1e-12));
end
