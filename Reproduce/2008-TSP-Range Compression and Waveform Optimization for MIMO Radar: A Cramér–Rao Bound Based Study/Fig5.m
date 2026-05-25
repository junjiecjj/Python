

clc;
clear all;
close all;

M = 10;
N = 10;
K = 2;
L = 256;
P = 1;
theta = [-16.5; -10];
b = [1; 20];
theta_jam = 5;
sigma2 = 1;
AINR_dB = 100;
AINR = 10^(AINR_dB/10);
angle_grid = -25:0.02:5;

radars(1).name = 'A(5,0.5)';
radars(1).dt = 5;
radars(1).dr = 0.5;
radars(2).name = 'B(0.5,0.5)';
radars(2).dt = 0.5;
radars(2).dr = 0.5;

B_all = cell(2,1);
R_all = cell(2,1);

for rr = 1:2
    dt = radars(rr).dt;
    dr = radars(rr).dr;
    fprintf('\n===== Radar %s =====\n', radars(rr).name);

    tx_pos = ((0:N-1) - (N-1)/2).' * dt;
    rx_pos = ((0:M-1) - (M-1)/2).' * dr;

    vfun = @(th) exp(1j*2*pi*tx_pos*sind(th));
    afun = @(th) exp(1j*2*pi*rx_pos*sind(th));
    vdfun = @(th) 1j*2*pi*tx_pos*cosd(th)*(pi/180).*exp(1j*2*pi*tx_pos*sind(th));
    adfun = @(th) 1j*2*pi*rx_pos*cosd(th)*(pi/180).*exp(1j*2*pi*rx_pos*sind(th));

    aj = afun(theta_jam);
    jammer_power = AINR * sigma2 / M;
    Q = sigma2 * eye(M) + jammer_power * (aj * aj');

    A = zeros(M,K);
    Ad = zeros(M,K);
    V = zeros(N,K);
    Vd = zeros(N,K);

    for k = 1:K
        A(:,k) = afun(theta(k));
        Ad(:,k) = adfun(theta(k));
        V(:,k) = vfun(theta(k));
        Vd(:,k) = vdfun(theta(k));
    end

    R_trace = design_trace_opt_two_target_target1(P, A, Ad, V, Vd, Q, b, L);
    R_all{rr} = R_trace;
    B_all{rr} = tx_beampattern(R_trace, vfun, angle_grid);
end

figure('Color','w','Position',[120 120 900 360]);

subplot(1,2,1);
plot(angle_grid, B_all{1}, 'b-', 'LineWidth', 1.5); grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(a) MIMO radar A(5,0.5)');
xlim([-20 0]);
ylim([-30 0]);
xline(theta(1), 'k--', 'LineWidth', 1.0);
xline(theta(2), 'k:', 'LineWidth', 1.0);
legend('Trace-Opt', '\theta_1', '\theta_2', 'Location', 'best');

subplot(1,2,2);
plot(angle_grid, B_all{2}, 'b-', 'LineWidth', 1.5); grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(b) MIMO radar B(0.5,0.5)');
xlim([-20 0]);
ylim([-30 0]);
xline(theta(1), 'k--', 'LineWidth', 1.0);
xline(theta(2), 'k:', 'LineWidth', 1.0);
legend('Trace-Opt', '\theta_1', '\theta_2', 'Location', 'best');

function R = design_trace_opt_two_target_target1(P, A, Ad, V, Vd, Q, b, L)
    N = size(V,1);
    K = length(b);
    dim = 3*K;
    idx_target1 = [1, K+1, 2*K+1];

    %% ========== Trace-Opt for target 1 using Appendix C ==========
    % Eq. (30): U = [V*(V'*V)^(-1/2), Vd*(Vd'*Vd)^(-1/2)].
    % Eq. (29): R = U*Lambda*U^H.
    % Eq. (23): minimize selected CRB diagonal entries by Schur complement.
    U = make_optimal_subspace(V, Vd);
    rU = size(U,2);

    cvx_begin sdp quiet
        variable Lambda(rU,rU) hermitian semidefinite
        variable u(3)
        Rcvx = U * Lambda * U';
        F = FIM_cvx_multi_paper(Rcvx, A, Ad, V, Vd, Q, b, L);
        minimize(sum(u))
        subject to
            trace(Rcvx) == P;
            for kk = 1:3
                e = zeros(dim,1);
                e(idx_target1(kk)) = 1;
                [F, e; e', u(kk)] >= 0;
            end
    cvx_end

    R = U * Lambda * U';
    R = hermitian_project(R);
    if size(R,1) ~= N
        error('R dimension is inconsistent with the number of transmit antennas.');
    end
    fprintf('Trace-Opt CVX status: %s\n', cvx_status);
end

function U = make_optimal_subspace(V, Vd)
    %% ========== Construct U according to Appendix C, eq. (30) ==========
    % The first block is V*(V'*V)^(-1/2), and the second block is Vd*(Vd'*Vd)^(-1/2).
    % This follows the paper's normalized subspace construction directly.
    Gv = V' * V;
    Gd = Vd' * Vd;
    Gv_inv_sqrt = matrix_inv_sqrt(Gv);
    Gd_inv_sqrt = matrix_inv_sqrt(Gd);
    U = [V * Gv_inv_sqrt, Vd * Gd_inv_sqrt];
end

function A_inv_sqrt = matrix_inv_sqrt(A)
    A = hermitian_project(A);
    [E,D] = eig(A);
    d = real(diag(D));
    tol = 1e-10 * max(1,max(d));
    d(d < tol) = tol;
    A_inv_sqrt = E * diag(1 ./ sqrt(d)) * E';
    A_inv_sqrt = hermitian_project(A_inv_sqrt);
end

function F = FIM_cvx_multi_paper(R, A, Ad, V, Vd, Q, b, L)
    K = length(b);
    b = b(:);

    A00 = A' / Q * A;
    Ad0 = Ad' / Q * A;
    A0d = A' / Q * Ad;
    Add = Ad' / Q * Ad;

    V00 = V' * R * V;
    Vd0 = Vd' * R * V;
    V0d = V' * R * Vd;
    Vdd = Vd' * R * Vd;

    Btt = conj(b) * b.';
    Bt = conj(b) * ones(1,K);

    F11c = L * Btt .* (Add .* V00 + Ad0 .* V0d + A0d .* Vd0 + A00 .* Vdd);
    F12c = L * Bt .* (Ad0 .* V00 + A00 .* Vd0);
    F22c = L * A00 .* V00;

    F11 = 2 * real(F11c);
    F12r = 2 * real(F12c);
    F12i = -2 * imag(F12c);
    F22rr = 2 * real(F22c);
    F22ri = -2 * imag(F22c);
    F22ii = 2 * real(F22c);

    F = [F11, F12r, F12i;
         F12r.', F22rr, F22ri;
         F12i.', F22ri.', F22ii];

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