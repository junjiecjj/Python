clear; clc; close all;

M = 10;
N = 10;
K = 2;
L = 256;
P = 1;
theta_true = [-16.5; -10];
b_true = [1; 20];
theta_jam = 5;
sigma2 = 1;
AINR_dB = 100;
AINR = 10^(AINR_dB/10);
err_grid = -3:0.1:3;

radars(1).name = 'A(5,0.5)';
radars(1).dt = 5;
radars(1).dr = 0.5;
radars(2).name = 'B(0.5,0.5)';
radars(2).dt = 0.5;
radars(2).dr = 0.5;

methods = {'Uncorrelated', 'Sum Beam', 'Trace Opt'};
num_radars = length(radars);
num_methods = length(methods);
num_err = length(err_grid);

RCRB_theta1 = zeros(num_radars, num_methods, num_err);
RCRB_b1 = zeros(num_radars, num_methods, num_err);

for rr = 1:num_radars
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

    [A_true, Ad_true, V_true, Vd_true] = build_steering_multi(theta_true, afun, adfun, vfun, vdfun);

    %% ========== Fixed uncorrelated waveform ==========
    % Uncorrelated waveform does not depend on the initial angle estimate, so its CRB is constant over err_grid.
    R_uncorr_fixed = P / N * eye(N);
    F_uncorr = FIM_numeric_multi_paper(R_uncorr_fixed, A_true, Ad_true, V_true, Vd_true, Q, b_true, L);
    C_uncorr = F_uncorr \ eye(3*K);
    RCRB_theta1(rr,1,:) = sqrt(max(real(C_uncorr(1,1)), 0));
    RCRB_b1(rr,1,:) = sqrt(max(real(C_uncorr(K+1,K+1) + C_uncorr(2*K+1,2*K+1)), 0));

    %% ========== Sum Beam and Trace-Opt with initial angle estimation error ==========
    % Only theta1 has initial estimation error; theta2 and both amplitudes are assumed exact.
    % Sum Beam and Trace-Opt are designed using theta_hat, then evaluated using true parameters.
    for ii = 1:num_err
        err = err_grid(ii);
        theta_hat = theta_true;
        theta_hat(1) = theta_true(1) + err;
        fprintf('Radar %s, error %.2f deg, %d / %d\n', radars(rr).name, err, ii, num_err);

        [A_hat, Ad_hat, V_hat, Vd_hat] = build_steering_multi(theta_hat, afun, adfun, vfun, vdfun);

        % Sum Beam points to the estimated target-of-interest angle theta1_hat, so it varies with err.
        v1_hat = V_hat(:,1);
        R_sum = P / real(v1_hat' * v1_hat) * (v1_hat * v1_hat');
        F_sum = FIM_numeric_multi_paper(R_sum, A_true, Ad_true, V_true, Vd_true, Q, b_true, L);
        C_sum = F_sum \ eye(3*K);
        RCRB_theta1(rr,2,ii) = sqrt(max(real(C_sum(1,1)), 0));
        RCRB_b1(rr,2,ii) = sqrt(max(real(C_sum(K+1,K+1) + C_sum(2*K+1,2*K+1)), 0));

        % Trace-Opt minimizes the target-1 CRB block using hatted parameters, then is evaluated at true parameters.
        R_trace = design_trace_opt_two_target_target1(P, A_hat, Ad_hat, V_hat, Vd_hat, Q, b_true, L);
        F_trace = FIM_numeric_multi_paper(R_trace, A_true, Ad_true, V_true, Vd_true, Q, b_true, L);
        C_trace = F_trace \ eye(3*K);
        RCRB_theta1(rr,3,ii) = sqrt(max(real(C_trace(1,1)), 0));
        RCRB_b1(rr,3,ii) = sqrt(max(real(C_trace(K+1,K+1) + C_trace(2*K+1,2*K+1)), 0));
    end
end

figure('Color','w','Position',[100 100 980 680]);

subplot(2,2,1);
plot_three_methods(err_grid, squeeze(RCRB_theta1(1,:,:)));
grid on; set(gca,'YScale','log');
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of \theta_1 (deg)');
title('(a) Root CRB of \theta_1 for MIMO radar A(5,0.5)');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

subplot(2,2,2);
plot_three_methods(err_grid, squeeze(RCRB_b1(1,:,:)));
grid on; set(gca,'YScale','log');
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of b_1');
title('(b) Root CRB of b_1 for MIMO radar A(5,0.5)');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

subplot(2,2,3);
plot_three_methods(err_grid, squeeze(RCRB_theta1(2,:,:)));
grid on; set(gca,'YScale','log');
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of \theta_1 (deg)');
title('(c) Root CRB of \theta_1 for MIMO radar B(0.5,0.5)');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

subplot(2,2,4);
plot_three_methods(err_grid, squeeze(RCRB_b1(2,:,:)));
grid on; set(gca,'YScale','log');
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of b_1');
title('(d) Root CRB of b_1 for MIMO radar B(0.5,0.5)');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

fprintf('\n===== Numerical ranges =====\n');
for rr = 1:num_radars
    fprintf('\nRadar %s\n', radars(rr).name);
    for mm = 1:num_methods
        theta_min = min(squeeze(RCRB_theta1(rr,mm,:)));
        theta_max = max(squeeze(RCRB_theta1(rr,mm,:)));
        b_min = min(squeeze(RCRB_b1(rr,mm,:)));
        b_max = max(squeeze(RCRB_b1(rr,mm,:)));
        fprintf('%s: theta1 RCRB [%.3e, %.3e], b1 RCRB [%.3e, %.3e]\n', methods{mm}, theta_min, theta_max, b_min, b_max);
    end
end

function [A, Ad, V, Vd] = build_steering_multi(theta, afun, adfun, vfun, vdfun)
    K = length(theta);
    M = length(afun(theta(1)));
    N = length(vfun(theta(1)));
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
end

function R = design_trace_opt_two_target_target1(P, A, Ad, V, Vd, Q, b, L)
    K = length(b);
    dim = 3*K;
    idx_target1 = [1, K+1, 2*K+1];

    %% ========== Trace-Opt for target 1 using Appendix C ==========
    % Appendix C shows that the optimal R lies in span{V,Vd}.
    % Write R = U*X*U', and apply the generalized Trace-Opt SDP in (23) to theta1, real(b1), and imag(b1).
    U = orth([V, Vd]);
    r = size(U,2);

    cvx_begin sdp quiet
        variable X(r,r) hermitian semidefinite
        variable u(3)
        Rcvx = U * X * U';
        F = FIM_cvx_multi_paper(Rcvx, A, Ad, V, Vd, Q, b, L);
        minimize(sum(u))
        subject to
            trace(X) == P;
            for kk = 1:3
                e = zeros(dim,1);
                e(idx_target1(kk)) = 1;
                [F, e; e', u(kk)] >= 0;
            end
    cvx_end

    R = U * X * U';
    R = hermitian_project(R);
    fprintf('Trace-Opt CVX status: %s\n', cvx_status);
end

function F = FIM_numeric_multi_paper(R, A, Ad, V, Vd, Q, b, L)
    K = length(b);
    b = b(:);
    R = hermitian_project(R);
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
    F = real((F + F')/2);
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

function R = hermitian_project(R)
    R = (R + R')/2;
end

function plot_three_methods(x, Y)
    hold on;
    plot(x, Y(1,:), 'b:', 'LineWidth', 1.5);
    plot(x, Y(2,:), 'g-.', 'LineWidth', 1.5);
    plot(x, Y(3,:), 'r-', 'LineWidth', 1.8);
end