%% ============================================================
%  Reproduce Fig. 3 in:
%  Jian Li et al., "Range Compression and Waveform Optimization
%  for MIMO Radar: A Cramer-Rao Bound Based Study"
%
%  Fig. 3:
%  MIMO Radar A(5, 0.5), single target
%  theta = -16.5 deg, b = 1
%  Four criteria:
%    (a) Angle-only
%    (b) Eigen-Opt
%    (c) Trace-Opt
%    (d) Det-Opt
%
%  Requirements:
%    CVX is needed for Eigen-Opt and Trace-Opt.
%    Det-Opt here avoids CVX log_det and uses 1-D search.
%% ============================================================

clear; clc; close all;

%% ========== Basic parameters ==========
M = 10;                 % number of transmit antennas
N = 10;                 % number of receive antennas
P = 1;                  % total transmit power

dt = 5;                 % transmit spacing, MIMO Radar A(5,0.5)
dr = 0.5;               % receive spacing

theta0 = -16.5;         % target angle, deg
theta_jam = 5;          % jammer angle, deg

b = 1;                  % target complex amplitude
sigma2 = 1;             % thermal noise variance

AINR_dB = 100;          % jammer array interference-to-noise ratio
AINR = 10^(AINR_dB/10);

angle_grid = -20:0.01:0;

%% ========== Array geometry ==========
% Use centered array reference point so that vdot^H v = 0.
tx_pos = ((0:M-1) - (M-1)/2).' * dt;
rx_pos = ((0:N-1) - (N-1)/2).' * dr;

% Steering vectors.
% Angle is in degrees.
a  = @(theta) exp(1j*2*pi*rx_pos*sind(theta));
v  = @(theta) exp(1j*2*pi*tx_pos*sind(theta));

% Derivatives with respect to theta in degrees.
% d/dtheta exp(j 2 pi d sin(theta)) =
% j 2 pi d cos(theta) * pi/180 * exp(...)
ad = @(theta) 1j*2*pi*rx_pos*cosd(theta)*(pi/180) .* exp(1j*2*pi*rx_pos*sind(theta));

vd = @(theta) 1j*2*pi*tx_pos*cosd(theta)*(pi/180) .* exp(1j*2*pi*tx_pos*sind(theta));

a0  = a(theta0);
v0  = v(theta0);
ad0 = ad(theta0);
vd0 = vd(theta0);

aj = a(theta_jam);

%% ========== Interference-plus-noise covariance Q ==========
% One strong spatial jammer plus white noise.
% This is the usual simulation model:
%
% Q = sigma^2 I + jammer_power * a_j a_j^H
%
% AINR is incident jammer power times N divided by sigma^2.
jammer_power = AINR * sigma2 / N;
Q = sigma2 * eye(N) + jammer_power * (aj * aj');

% Use inverse through backslash when possible.
Qinv = inv(Q);

%% ========== Angle-only design ==========
% For the single-target case, paper Appendix D gives the structure:
%
% R = zeta * P/||v||^2 * v v^H
%   + (1-zeta) * P/||vdot||^2 * vdot vdot^H
%
% For MIMO Radar A(5,0.5), alpha < beta, so use very small zeta.
zeta = 1e-4;

R_angle = zeta * P/(v0' * v0) * (v0 * v0') ...
        + (1-zeta) * P/(vd0' * vd0) * (vd0 * vd0');

R_angle = hermitian_project(R_angle);

%% ========== Eigen-Opt by CVX ==========
% Maximize the minimum eigenvalue of the Fisher information matrix.
%
% Equivalent to:
%   maximize t
%   subject to FIM(R) - t I >= 0
%              trace(R) = P
%              R >= 0

cvx_begin sdp quiet
    variable R_eigen(M,M) hermitian semidefinite
    variable t_eigen

    F_eigen = FIM_cvx_single(R_eigen, a0, ad0, v0, vd0, Qinv, b);

    maximize(t_eigen)
    subject to
        trace(R_eigen) == P;
        F_eigen - t_eigen * eye(3) >= 0;
cvx_end

R_eigen = hermitian_project(R_eigen);

%% ========== Trace-Opt by CVX ==========
% Minimize trace(inv(FIM)).
%
% SDP form:
%   minimize sum u_k
%   subject to [F e_k; e_k' u_k] >= 0

cvx_begin sdp quiet
    variable R_trace(M,M) hermitian semidefinite
    variable u_trace(3)

    F_trace = FIM_cvx_single(R_trace, a0, ad0, v0, vd0, Qinv, b);

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

%% ========== Det-Opt without CVX log_det ==========
% Avoid:
%   maximize(log_det(F))
%
% because some CVX installations fail due to vec/det_rootn path conflict.
%
% In the single-target case, the optimal R lies in span{v, vdot}.
% Since the centered array gives vdot^H v = 0, search:
%
% R(lambda) = lambda * u1 u1^H + (P-lambda) * u2 u2^H
%
% where:
%   u1 = v / ||v||
%   u2 = orthogonalized vdot / ||vdot||

u1 = v0 / norm(v0);

u2 = vd0 - u1 * (u1' * vd0);
u2 = u2 / norm(u2);

det_objective = @(lambda1) -logdet_numeric( ...
    FIM_numeric_single( ...
        lambda1 * (u1*u1') + (P-lambda1) * (u2*u2'), ...
        a0, ad0, v0, vd0, Qinv, b ...
    ) ...
);

lambda1_opt = fminbnd(det_objective, 1e-8, P-1e-8);
lambda2_opt = P - lambda1_opt;

R_det = lambda1_opt * (u1*u1') + lambda2_opt * (u2*u2');
R_det = hermitian_project(R_det);

fprintf('\nDet-Opt lambda1 = %.6g, lambda2 = %.6g\n', ...
    lambda1_opt, lambda2_opt);

%% ========== Compute transmit beampatterns ==========
B_angle = tx_beampattern(R_angle, v, angle_grid);
B_eigen = tx_beampattern(R_eigen, v, angle_grid);
B_trace = tx_beampattern(R_trace, v, angle_grid);
B_det   = tx_beampattern(R_det,   v, angle_grid);

%% ========== Plot Fig. 3 style ==========
figure('Color','w');

subplot(2,2,1);
plot(angle_grid, B_angle, 'LineWidth', 1.2);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(a) Angle-only');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,2);
plot(angle_grid, B_eigen, 'LineWidth', 1.2);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(b) Eigen-Opt');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,3);
plot(angle_grid, B_trace, 'LineWidth', 1.2);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(c) Trace-Opt');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,4);
plot(angle_grid, B_det, 'LineWidth', 1.2);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(d) Det-Opt');
xlim([-20 0]);
ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

%% ========== Optional: display CRB values ==========
F_angle = FIM_numeric_single(R_angle, a0, ad0, v0, vd0, Qinv, b);
F_eigen_num = FIM_numeric_single(R_eigen, a0, ad0, v0, vd0, Qinv, b);
F_trace_num = FIM_numeric_single(R_trace, a0, ad0, v0, vd0, Qinv, b);
F_det_num   = FIM_numeric_single(R_det,   a0, ad0, v0, vd0, Qinv, b);

fprintf('\nRoot CRB of theta, in degrees:\n');
fprintf('Angle-only : %.6e\n', sqrt(real(inv(F_angle(1,1)))));
fprintf('Eigen-Opt  : %.6e\n', sqrt(real(inv(F_eigen_num(1,1)))));
fprintf('Trace-Opt  : %.6e\n', sqrt(real(inv(F_trace_num(1,1)))));
fprintf('Det-Opt    : %.6e\n', sqrt(real(inv(F_det_num(1,1)))));

%% ============================================================
%  Local functions
%% ============================================================

function F = FIM_numeric_single(R, a0, ad0, v0, vd0, Qinv, b)
    % Numeric 3 x 3 real Fisher information matrix.
    %
    % Unknown real parameter vector:
    %   eta = [theta, real(b), imag(b)]^T
    %
    % This function uses the Gaussian mean-derivative formula:
    %
    %   F_ij = 2 Re{ tr( dM_i^H Q^{-1} dM_j ) }
    %
    % where the waveform covariance is R = S S^H.
    %
    % Single-target mean:
    %   M = b * a(theta) * v(theta)^T S
    %
    % The derivatives are represented through R, so S itself is not needed.

    R = hermitian_project(R);

    A00 = a0'  * Qinv * a0;
    A10 = ad0' * Qinv * a0;
    A01 = a0'  * Qinv * ad0;
    A11 = ad0' * Qinv * ad0;

    V00 = v0'  * R * v0;
    V10 = vd0' * R * v0;
    V01 = v0'  * R * vd0;
    V11 = vd0' * R * vd0;

    % derivative wrt theta:
    % dM/dtheta = b * adot * v^T S + b * a * vdot^T S
    F11 = 2 * real( abs(b)^2 * ...
        ( A11 * V00 + A10 * V01 + A01 * V10 + A00 * V11 ) );

    % derivative wrt real(b):
    % dM/db_R = a v^T S
    %
    % derivative wrt imag(b):
    % dM/db_I = j a v^T S

    % Cross term between theta and real(b)
    G_theta_b = conj(b) * (A10 * V00 + A00 * V10);

    F12 = 2 * real(G_theta_b);
    F13 = 2 * real(1j * G_theta_b);

    F22 = 2 * real(A00 * V00);
    F23 = 0;
    F33 = F22;

    F = [F11, F12, F13;
         F12, F22, F23;
         F13, F23, F33];

    F = real((F + F')/2);
end


function F = FIM_cvx_single(R, a0, ad0, v0, vd0, Qinv, b)
    % CVX-compatible Fisher information matrix.
    %
    % Keep all expressions affine in R.

    A00 = a0'  * Qinv * a0;
    A10 = ad0' * Qinv * a0;
    A01 = a0'  * Qinv * ad0;
    A11 = ad0' * Qinv * ad0;

    V00 = v0'  * R * v0;
    V10 = vd0' * R * v0;
    V01 = v0'  * R * vd0;
    V11 = vd0' * R * vd0;

    F11 = 2 * real( abs(b)^2 * ...
        ( A11 * V00 + A10 * V01 + A01 * V10 + A00 * V11 ) );

    G_theta_b = conj(b) * (A10 * V00 + A00 * V10);

    F12 = 2 * real(G_theta_b);
    F13 = 2 * real(1j * G_theta_b);

    F22 = 2 * real(A00 * V00);
    F23 = 0;
    F33 = F22;

    F = [F11, F12, F13;
         F12, F22, F23;
         F13, F23, F33];
end


function B_dB = tx_beampattern(R, vfun, angle_grid)
    % Transmit beampattern:
    %
    %   B(theta) = v(theta)^H R v(theta)

    R = hermitian_project(R);

    B = zeros(size(angle_grid));

    for ii = 1:length(angle_grid)
        vv = vfun(angle_grid(ii));
        B(ii) = real(vv' * R * vv);
    end

    B = B ./ max(B);
    B_dB = 10 * log10(B + eps);
end


function y = logdet_numeric(F)
    % Stable log-det for positive semidefinite numeric matrix.

    F = real((F + F')/2);
    ev = eig(F);
    ev = max(real(ev), 1e-12);
    y = sum(log(ev));
end


function R = hermitian_project(R)
    % Numerical Hermitian projection.

    R = (R + R')/2;
end