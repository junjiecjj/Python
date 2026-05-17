%% ============================================================
%  Reproduce Fig. 3 strictly following the paper formulas
%
%  Jian Li et al.,
%  "Range Compression and Waveform Optimization for MIMO Radar:
%   A Cramer-Rao Bound Based Study"
%
%  Fig. 3:
%    MIMO Radar A(5,0.5)
%    single target: theta = -16.5 deg, b = 1
%
%  Criteria:
%    (a) Angle-only: closed-form, Appendix D, eqs. (39)/(40)
%    (b) Eigen-Opt : SDP, eq. (27)
%    (c) Trace-Opt : SDP, eqs. (22)/(23)
%    (d) Det-Opt   : closed-form, Appendix E, eq. (42)
%
%  Important:
%    The paper model is X = b*a(theta)*v(theta)^T*S + Z.
%    Therefore transmit-side quadratic form is:
%       v(theta)^T R v(theta)^*
%    not v(theta)^H R v(theta).
%
%  Requirement:
%    CVX is needed for Eigen-Opt and Trace-Opt.
%% ============================================================

clear; clc; close all;

%% ========== Parameters ==========
M = 10;                 % transmit antennas
N = 10;                 % receive antennas
L = 256;                % waveform length, as used in the paper
P = 1;                  % total transmit power, trace(R)=P

dt = 5;                 % MIMO Radar A transmit spacing
dr = 0.5;               % MIMO Radar A receive spacing

theta0 = -16.5;         % target angle, deg
theta_jam = 5;          % jammer angle, deg

b = 1;                  % target complex amplitude
sigma2 = 1;             % thermal noise variance

AINR_dB = 100;          % jammer array interference-to-noise ratio
AINR = 10^(AINR_dB/10);

angle_grid = -20:0.01:0;

%% ========== Array geometry ==========
% Centered reference point so that vdot^T v^* = 0 approximately.
tx_pos = ((0:M-1) - (M-1)/2).' * dt;
rx_pos = ((0:N-1) - (N-1)/2).' * dr;

% Steering vectors. Angle is in degrees.
vfun = @(th) exp(1j*2*pi*tx_pos*sind(th));
afun = @(th) exp(1j*2*pi*rx_pos*sind(th));

% Derivatives wrt theta in degrees.
% pi/180 appears because theta is in degrees.
vdfun = @(th) 1j*2*pi*tx_pos*cosd(th)*(pi/180) .* exp(1j*2*pi*tx_pos*sind(th));

adfun = @(th) 1j*2*pi*rx_pos*cosd(th)*(pi/180) .* exp(1j*2*pi*rx_pos*sind(th));

v0  = vfun(theta0);
vd0 = vdfun(theta0);
a0  = afun(theta0);
ad0 = adfun(theta0);

%% ========== Interference-plus-noise covariance Q ==========
aj = afun(theta_jam);

% AINR = incident jammer power * N / sigma2
jammer_power = AINR * sigma2 / N;

Q = sigma2 * eye(N) + jammer_power * (aj * aj');
Qinv = inv(Q);

%% ========== Common alpha and beta ==========
% These correspond to the single-target closed-form derivations
% in Appendix D and Appendix E.
%
% alpha = |b|^2 ||v||^2 * (ad^H Q^{-1} ad
%                         - |ad^H Q^{-1} a|^2/(a^H Q^{-1}a))
%
% beta  = |b|^2 ||vd||^2 * (a^H Q^{-1} a)
%
% The common L factor does not affect comparisons, but the FIM itself
% uses L explicitly.

A00 = real(a0' /Q * a0);
A10 = ad0' /Q * a0;
A11 = real(ad0' /Q * ad0);

nv  = real(v0'  * v0);
nvd = real(vd0' * vd0);

alpha = abs(b)^2 * nv  * real(A11 - abs(A10)^2 / A00);
beta  = abs(b)^2 * nvd * A00;

fprintf('\nalpha = %.6e\n', alpha);
fprintf('beta  = %.6e\n', beta);
fprintf('beta/3 = %.6e\n\n', beta/3);

%% ============================================================
%  1) Angle-only, Appendix D, eqs. (39)/(40)
%% ============================================================

tol_ab = 1e-10 * max(1, max(abs(alpha), abs(beta)));

if alpha > beta + tol_ab
    % Eq. (39): pure sum-beam component.
    % Because the model uses v^T S, the covariance component is
    % conj(v)*v^T / ||v||^2.
    R_angle = P / nv * (v0 * v0');
    fprintf('Angle-only uses eq. (39): pure sum-beam.\n');
elseif beta > alpha + tol_ab
    % Eq. (40): almost pure difference-beam component.
    % zeta should be small and positive.
    zeta = 1e-4;
    R_angle = zeta * P / nv  * (v0 * v0') + (1-zeta) * P / nvd * (vd0 * vd0');
    fprintf('Angle-only uses eq. (40): zeta = %.2e.\n', zeta);
else
    % alpha approximately equals beta.
    % Any split is optimal; choose 50/50.
    zeta = 0.5;
    R_angle = zeta * P / nv  * (v0 * v0') + (1-zeta) * P / nvd * (vd0 * vd0');
    fprintf('Angle-only: alpha approx beta, use 50/50 split.\n');
end
R_angle = hermitian_project(R_angle);

%% ============================================================
%  2) Eigen-Opt, paper eq. (27)
%
%  maximize t
%  subject to F(R) - tI >= 0
%             trace(R) = P
%             R >= 0
%% ============================================================

cvx_begin sdp quiet
    variable R_eigen(M,M) hermitian semidefinite
    variable t_eigen

    F_eigen = FIM_cvx_single_paper(R_eigen, a0, ad0, v0, vd0, Q, b, L);

    maximize(t_eigen)
    subject to
        trace(R_eigen) == P;
        F_eigen - t_eigen * eye(3) >= 0;
cvx_end

R_eigen = hermitian_project(R_eigen);

fprintf('Eigen-Opt CVX status: %s\n', cvx_status);

%% ============================================================
%  3) Trace-Opt, paper eqs. (22)/(23)
%
%  minimize trace(inv(F))
%  SDP epigraph:
%     minimize sum u_k
%     subject to [F e_k; e_k^T u_k] >= 0
%% ============================================================

cvx_begin sdp quiet
    variable R_trace(M,M) hermitian semidefinite
    variable u_trace(3)
    F_trace = FIM_cvx_single_paper(R_trace, a0, ad0, v0, vd0, Q, b, L);

    minimize(sum(u_trace))
    subject to
        trace(R_trace) == P;

        for k = 1:3
            ek = zeros(3,1);
            ek(k) = 1;
            [F_trace, ek; ek', u_trace(k)] >= 0;
        end
cvx_end
R_trace = hermitian_project(R_trace);
fprintf('Trace-Opt CVX status: %s\n', cvx_status);

%% ============================================================
%  4) Det-Opt, Appendix E, eq. (42)
%
%  If alpha >= beta/3:
%      lambda1 = P, lambda2 = 0
%  Else:
%      lambda1 = 2*beta*P / (3*(beta-alpha))
%      lambda2 = P - lambda1
%% ============================================================
if alpha >= beta/3
    lambda1 = P;
    lambda2 = 0;
    fprintf('Det-Opt uses eq. (42), case alpha >= beta/3.\n');
else
    lambda1 = 2 * beta * P / (3 * (beta - alpha));
    lambda2 = P - lambda1;
    fprintf('Det-Opt uses eq. (42), case alpha < beta/3.\n');
end
R_det = lambda1 / nv  * (v0 * v0') + lambda2 / nvd * (vd0 * vd0');

R_det = hermitian_project(R_det);

fprintf('Det-Opt lambda1 = %.6e, lambda2 = %.6e\n', lambda1, lambda2);

%% ========== Beampatterns ==========
B_angle = tx_beampattern_paper(R_angle, vfun, angle_grid);
B_eigen = tx_beampattern_paper(R_eigen, vfun, angle_grid);
B_trace = tx_beampattern_paper(R_trace, vfun, angle_grid);
B_det   = tx_beampattern_paper(R_det,   vfun, angle_grid);

%% ========== Plot Fig. 3 style ==========
figure('Color','w', 'Position', [100 100 900 650]);

subplot(2,2,1);
plot(angle_grid, B_angle, 'LineWidth', 1.3);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(a) Angle-only');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,2);
plot(angle_grid, B_eigen, 'LineWidth', 1.3);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(b) Eigen-Opt');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,3);
plot(angle_grid, B_trace, 'LineWidth', 1.3);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(c) Trace-Opt');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,4);
plot(angle_grid, B_det, 'LineWidth', 1.3);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(d) Det-Opt');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

%% ========== Optional: CRB check ==========
F_angle_num = FIM_numeric_single_paper(R_angle, a0, ad0, v0, vd0, Q, b, L);
F_eigen_num = FIM_numeric_single_paper(R_eigen, a0, ad0, v0, vd0, Q, b, L);
F_trace_num = FIM_numeric_single_paper(R_trace, a0, ad0, v0, vd0, Q, b, L);
F_det_num   = FIM_numeric_single_paper(R_det,   a0, ad0, v0, vd0, Q, b, L);

C_angle = inv(F_angle_num);
C_eigen = inv(F_eigen_num);
C_trace = inv(F_trace_num);
C_det   = inv(F_det_num);

fprintf('\nRoot CRB of theta, in degrees:\n');
fprintf('Angle-only : %.6e\n', sqrt(real(C_angle(1,1))));
fprintf('Eigen-Opt  : %.6e\n', sqrt(real(C_eigen(1,1))));
fprintf('Trace-Opt  : %.6e\n', sqrt(real(C_trace(1,1))));
fprintf('Det-Opt    : %.6e\n', sqrt(real(C_det(1,1))));

%% ============================================================
%  Local functions
%% ============================================================

function F = FIM_numeric_single_paper(R, a0, ad0, v0, vd0, Q, b, L)
    % Single-target real FIM following the paper model:
    %   X = b*a(theta)*v(theta)^T*S + Z
    % Parameter vector:
    %   eta = [theta, real(b), imag(b)]^T
    % The paper uses R = S*S^H/L, hence every FIM block has factor L.
    % Since the signal uses v^T S, the transmit-side inner product is:
    %
    %   y_q^T R y_p^*
    % implemented as:
    %   y_q.' * R * conj(y_p)

    R = hermitian_project(R);

    %% Receive-side terms
    A00 = a0'  /Q * a0;
    Ad0 = ad0' /Q * a0;
    A0d = a0'  /Q * ad0;
    Add = ad0' /Q * ad0;

    %% Transmit-side terms
    T_v_v   = v0.'  * R * conj(v0);
    T_v_vd  = v0.'  * R * conj(vd0);
    T_vd_v  = vd0.' * R * conj(v0);
    T_vd_vd = vd0.' * R * conj(vd0);

    %% theta-theta block, corresponding to F11
    Gtt = abs(b)^2 * ...
        ( Add * T_v_v ...
        + Ad0 * T_vd_v ...
        + A0d * T_v_vd ...
        + A00 * T_vd_vd );

    F11 = 2 * L * real(Gtt);

    %% theta-amplitude block, corresponding to F12 after real split
    Gtr = conj(b) * ...
        ( Ad0 * T_v_v ...
        + A00 * T_v_vd );

    Gti = 1j * Gtr;

    F12 = 2 * L * real(Gtr);
    F13 = 2 * L * real(Gti);

    %% amplitude-amplitude block, corresponding to F22 after real split
    Grr = A00 * T_v_v;

    F22 = 2 * L * real(Grr);
    F23 = 2 * L * real(1j * Grr);
    F33 = 2 * L * real(Grr);

    F = [F11, F12, F13;
         F12, F22, F23;
         F13, F23, F33];

    F = real((F + F')/2);
end


function F = FIM_cvx_single_paper(R, a0, ad0, v0, vd0, Q, b, L)
    % CVX-compatible single-target real FIM.
    %
    % Same as FIM_numeric_single_paper, but all terms are affine in R.

    %% Receive-side terms
    A00 = a0' /Q * a0;
    Ad0 = ad0' /Q * a0;
    A0d = a0'  /Q * ad0;
    Add = ad0' /Q * ad0;

    %% Transmit-side terms
    T_v_v   = v0.'  * R * conj(v0);
    T_v_vd  = v0.'  * R * conj(vd0);
    T_vd_v  = vd0.' * R * conj(v0);
    T_vd_vd = vd0.' * R * conj(vd0);

    %% F11
    Gtt = abs(b)^2 * ...
        ( Add * T_v_v ...
        + Ad0 * T_vd_v ...
        + A0d * T_v_vd ...
        + A00 * T_vd_vd );

    F11 = 2 * L * real(Gtt);

    %% F12, split into real(b), imag(b)
    Gtr = conj(b) * ...
        ( Ad0 * T_v_v ...
        + A00 * T_v_vd );

    Gti = 1j * Gtr;

    F12 = 2 * L * real(Gtr);
    F13 = 2 * L * real(Gti);

    %% F22, split into real/imag amplitude block
    Grr = A00 * T_v_v;

    F22 = 2 * L * real(Grr);
    F23 = 2 * L * real(1j * Grr);
    F33 = 2 * L * real(Grr);

    F = [F11, F12, F13;
         F12, F22, F23;
         F13, F23, F33];

    F = 0.5 * (F + F.');
end


function B_dB = tx_beampattern_paper(R, vfun, angle_grid)
    % Transmit beampattern for paper model:
    %   B(theta) = v(theta)^T R v(theta)^*
    % not v^H R v.

    R = hermitian_project(R);
    B = zeros(size(angle_grid));
    for ii = 1:length(angle_grid)
        vv = vfun(angle_grid(ii));
        B(ii) = real(vv.' * R * conj(vv));
    end

    B = B ./ max(B);
    B_dB = 10 * log10(B + eps);
end


function R = hermitian_project(R)
    R = (R + R')/2;
end