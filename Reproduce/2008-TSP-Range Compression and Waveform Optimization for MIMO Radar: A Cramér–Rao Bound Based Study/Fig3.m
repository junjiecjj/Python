clear; clc; close all;

M = 10; 
N = 10; 
L = 256; 
P = 1;
dt = 5; 
dr = 0.5;
theta0 = -16.5; 
theta_jam = 5;
b = 1; 
sigma2 = 1;
AINR_dB = 100; 
AINR = 10^(AINR_dB/10);
angle_grid = -20:0.01:0;

tx_pos = ((0:M-1) - (M-1)/2).' * dt;
rx_pos = ((0:N-1) - (N-1)/2).' * dr;

vfun = @(th) exp(1j*2*pi*tx_pos*sind(th));
afun = @(th) exp(1j*2*pi*rx_pos*sind(th));
vdfun = @(th) 1j*2*pi*tx_pos*cosd(th)*(pi/180).*exp(1j*2*pi*tx_pos*sind(th));
adfun = @(th) 1j*2*pi*rx_pos*cosd(th)*(pi/180).*exp(1j*2*pi*rx_pos*sind(th));

v0 = vfun(theta0); 
vd0 = vdfun(theta0);
a0 = afun(theta0); 
ad0 = adfun(theta0);
aj = afun(theta_jam);

jammer_power = AINR * sigma2 / N;
Q = sigma2 * eye(N) + jammer_power * (aj * aj');

A00 = real(a0' / Q * a0);
A10 = ad0' / Q * a0;
A11 = real(ad0' / Q * ad0);
nv = real(v0' * v0);
nvd = real(vd0' * vd0);

%% ============================================================
%  1) Angle-only, Appendix D, eqs. (39)/(40)
%% ============================================================

alpha = L * abs(b)^2 * nv * real(A11 - abs(A10)^2 / A00);
beta = L * abs(b)^2 * nvd * A00;
fprintf('alpha = %.6e, beta = %.6e, beta/3 = %.6e\n', alpha, beta, beta/3);

tol_ab = 1e-10 * max(1, max(abs(alpha), abs(beta)));

if alpha > beta + tol_ab
    R_angle = P / nv * (v0 * v0');
    fprintf('Angle-only: use equation (39).\n');
elseif beta > alpha + tol_ab
    zeta = 1e-4;
    R_angle = zeta * P / nv * (v0 * v0') + (1-zeta) * P / nvd * (vd0 * vd0');
    fprintf('Angle-only: use equation (40), zeta = %.2e.\n', zeta);
else
    zeta = 0.5;
    R_angle = zeta * P / nv * (v0 * v0') + (1-zeta) * P / nvd * (vd0 * vd0');
    fprintf('Angle-only: alpha approximately equals beta, use 50/50 split.\n');
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
            ek = zeros(3,1); ek(k) = 1;
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
    lambda1 = P; lambda2 = 0;
    fprintf('Det-Opt: use equation (42), alpha >= beta/3.\n');
else
    lambda1 = 2 * beta * P / (3 * (beta - alpha));
    lambda2 = P - lambda1;
    fprintf('Det-Opt: use equation (42), alpha < beta/3.\n');
end
R_det = lambda1 / nv * (v0 * v0') + lambda2 / nvd * (vd0 * vd0');
R_det = hermitian_project(R_det);
fprintf('Det-Opt lambda1 = %.6e, lambda2 = %.6e\n', lambda1, lambda2);

%% ========== Beampatterns ==========
B_angle = tx_beampattern(R_angle, vfun, angle_grid);
B_eigen = tx_beampattern(R_eigen, vfun, angle_grid);
B_trace = tx_beampattern(R_trace, vfun, angle_grid);
B_det = tx_beampattern(R_det, vfun, angle_grid);

figure('Color','w','Position',[100 100 900 650]);

subplot(2,2,1);
plot(angle_grid, B_angle, 'LineWidth', 1.3); grid on;
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(a) Angle-only'); xlim([-20 0]); ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,2);
plot(angle_grid, B_eigen, 'LineWidth', 1.3); grid on;
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(b) Eigen-Opt'); xlim([-20 0]); ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,3);
plot(angle_grid, B_trace, 'LineWidth', 1.3); grid on;
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(c) Trace-Opt'); xlim([-20 0]); ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

subplot(2,2,4);
plot(angle_grid, B_det, 'LineWidth', 1.3); grid on;
xlabel('Angle (deg)'); ylabel('Beampattern (dB)');
title('(d) Det-Opt'); xlim([-20 0]); ylim([-30 0]);
xline(theta0, 'k--', 'LineWidth', 1.0);

F_angle_num = FIM_numeric_single_paper(R_angle, a0, ad0, v0, vd0, Q, b, L);
F_eigen_num = FIM_numeric_single_paper(R_eigen, a0, ad0, v0, vd0, Q, b, L);
F_trace_num = FIM_numeric_single_paper(R_trace, a0, ad0, v0, vd0, Q, b, L);
F_det_num = FIM_numeric_single_paper(R_det, a0, ad0, v0, vd0, Q, b, L);

C_angle = F_angle_num \ eye(3);
C_eigen = F_eigen_num \ eye(3);
C_trace = F_trace_num \ eye(3);
C_det = F_det_num \ eye(3);

fprintf('\nRoot CRB of theta, in degrees:\n');
fprintf('Angle-only : %.6e\n', sqrt(real(C_angle(1,1))));
fprintf('Eigen-Opt  : %.6e\n', sqrt(real(C_eigen(1,1))));
fprintf('Trace-Opt  : %.6e\n', sqrt(real(C_trace(1,1))));
fprintf('Det-Opt    : %.6e\n', sqrt(real(C_det(1,1))));

function F = FIM_numeric_single_paper(R, a0, ad0, v0, vd0, Q, b, L)
    R = hermitian_project(R);
    A00 = a0' / Q * a0;
    Ad0 = ad0' / Q * a0;
    A0d = a0' / Q * ad0;
    Add = ad0' / Q * ad0;
    V00 = v0' * R * v0;
    Vd0 = vd0' * R * v0;
    V0d = v0' * R * vd0;
    Vdd = vd0' * R * vd0;
    F11c = L * abs(b)^2 * (Add * V00 + Ad0 * V0d + A0d * Vd0 + A00 * Vdd);
    F12c = L * conj(b) * (Ad0 * V00 + A00 * Vd0);
    F22c = L * A00 * V00;
    F = 2 * [real(F11c), real(F12c), -imag(F12c);
             real(F12c), real(F22c), -imag(F22c);
            -imag(F12c), imag(F22c),  real(F22c)];
    F = real((F + F')/2);
end

function F = FIM_cvx_single_paper(R, a0, ad0, v0, vd0, Q, b, L)
    A00 = a0' / Q * a0;
    Ad0 = ad0' / Q * a0;
    A0d = a0' / Q * ad0;
    Add = ad0' / Q * ad0;
    V00 = v0' * R * v0;
    Vd0 = vd0' * R * v0;
    V0d = v0' * R * vd0;
    Vdd = vd0' * R * vd0;
    F11c = L * abs(b)^2 * (Add * V00 + Ad0 * V0d + A0d * Vd0 + A00 * Vdd);
    F12c = L * conj(b) * (Ad0 * V00 + A00 * Vd0);
    F22c = L * A00 * V00;
    F = 2 * [real(F11c), real(F12c), -imag(F12c);
             real(F12c), real(F22c), -imag(F22c);
            -imag(F12c), imag(F22c),  real(F22c)];
    F = 0.5 * (F + F.');
end

function B_dB = tx_beampattern(R, vfun, angle_grid)
    R = hermitian_project(R);
    B = zeros(size(angle_grid));
    for ii = 1:length(angle_grid)
        vv = vfun(angle_grid(ii));
        B(ii) = real(vv' * R * vv);
    end
    B = B / max(B);
    B_dB = 10 * log10(B + eps);
end

function R = hermitian_project(R)
    R = (R + R')/2;
end