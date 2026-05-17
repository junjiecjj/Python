

clc;
clear all;
close all;

M = 10;
N = 10;
L = 256;
P = 1;
theta_true = -16.5;
theta_jam = 5;
b_true = 1;
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
radars(3).name = 'C(0.5,5)';
radars(3).dt = 0.5;
radars(3).dr = 5;

methods = {'Uncorr', 'Sum', 'Angle', 'Eigen', 'Trace', 'Det'};
num_radars = length(radars);
num_methods = length(methods);
num_err = length(err_grid);

RCRB_theta = zeros(num_radars, num_methods, num_err);
RCRB_b = zeros(num_radars, num_methods, num_err);

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

    a_true = afun(theta_true);
    ad_true = adfun(theta_true);
    v_true = vfun(theta_true);
    vd_true = vdfun(theta_true);

    for ii = 1:num_err
        err = err_grid(ii);
        theta_hat = theta_true + err;
        fprintf('Radar %s, error %.2f deg, %d / %d\n', radars(rr).name, err, ii, num_err);

        a_hat = afun(theta_hat);
        ad_hat = adfun(theta_hat);
        v_hat = vfun(theta_hat);
        vd_hat = vdfun(theta_hat);

        R_uncorr = P / N * eye(N);
        R_sum = P / real(v_hat' * v_hat) * (v_hat * v_hat');
        R_angle = design_angle_only(P, a_hat, ad_hat, v_hat, vd_hat, Q, b_true, L);
        R_eigen = design_eigen_opt(P, a_hat, ad_hat, v_hat, vd_hat, Q, b_true, L);
        R_trace = design_trace_opt(P, a_hat, ad_hat, v_hat, vd_hat, Q, b_true, L);
        R_det = design_det_opt(P, a_hat, ad_hat, v_hat, vd_hat, Q, b_true, L);

        R_list = {R_uncorr, R_sum, R_angle, R_eigen, R_trace, R_det};

        for mm = 1:num_methods
            R_now = hermitian_project(R_list{mm});
            F_true = FIM_numeric_single(R_now, a_true, ad_true, v_true, vd_true, Q, b_true, L);
            C_true = F_true \ eye(3);
            RCRB_theta(rr,mm,ii) = sqrt(max(real(C_true(1,1)), 0));
            RCRB_b(rr,mm,ii) = sqrt(max(real(C_true(2,2) + C_true(3,3)), 0));
        end
    end
end

%% ========== Plot Fig. 4 style ==========
figure('Color','w', 'Position', [100 100 1100 750]);

% Line styles similar to the paper.
lineStyles = {':', '-.', '--', 's', '-', 'o'};
lineWidths = [1.1, 1.1, 1.1, 1.0, 1.4, 1.0];

% Use markers sparsely for Eigen and Det.
markerEvery = 5;

% -------- Radar A --------
subplot(3,2,1);
plot_all_methods(err_grid, squeeze(RCRB_theta(1,:,:)), methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Angle (deg)');
title('(a) Root CRB of \theta for MIMO radar A(5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-3 1e0]);

subplot(3,2,2);
plot_all_methods(err_grid, squeeze(RCRB_b(1,:,:)), methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Amplitude');
title('(b) Root CRB of b for MIMO radar A(5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-2 1e1]);

% -------- Radar B --------
subplot(3,2,3);
plot_all_methods(err_grid, squeeze(RCRB_theta(2,:,:)), methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Angle (deg)');
title('(c) Root CRB of \theta for MIMO radar B(0.5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-3 1e0]);

subplot(3,2,4);
plot_all_methods(err_grid, squeeze(RCRB_b(2,:,:)), methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Amplitude');
title('(d) Root CRB of b for MIMO radar B(0.5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-2 1e1]);

% -------- Radar C --------
subplot(3,2,5);
plot_all_methods(err_grid, squeeze(RCRB_theta(3,:,:)), methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Angle (deg)');
title('(e) Root CRB of \theta for MIMO radar C(0.5,5)');
set(gca, 'YScale', 'log');
ylim([1e-3 1e0]);

subplot(3,2,6);
plot_all_methods(err_grid, squeeze(RCRB_b(3,:,:)), methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Amplitude');
title('(f) Root CRB of b for MIMO radar C(0.5,5)');
set(gca, 'YScale', 'log');
ylim([1e-2 1e1]);


fprintf('\n===== Numerical ranges =====\n');
for rr = 1:num_radars
    fprintf('\nRadar %s\n', radars(rr).name);
    for mm = 1:num_methods
        fprintf('%s: theta RCRB [%.3e, %.3e], b RCRB [%.3e, %.3e]\n', methods{mm}, min(squeeze(RCRB_theta(rr,mm,:))), max(squeeze(RCRB_theta(rr,mm,:))), min(squeeze(RCRB_b(rr,mm,:))), max(squeeze(RCRB_b(rr,mm,:))));
    end
end

function R = design_angle_only(P, a0, ad0, v0, vd0, Q, b, L)
    A00 = real(a0' / Q * a0);
    A10 = ad0' / Q * a0;
    A11 = real(ad0' / Q * ad0);
    nv = real(v0' * v0);
    nvd = real(vd0' * vd0);
    alpha = L * abs(b)^2 * nv * real(A11 - abs(A10)^2 / A00);
    beta = L * abs(b)^2 * nvd * A00;
    tol_ab = 1e-10 * max(1, max(abs(alpha), abs(beta)));
    if alpha > beta + tol_ab
        R = P / nv * (v0 * v0');
    elseif beta > alpha + tol_ab
        zeta = 1e-4;
        R = zeta * P / nv * (v0 * v0') + (1-zeta) * P / nvd * (vd0 * vd0');
    else
        zeta = 0.5;
        R = zeta * P / nv * (v0 * v0') + (1-zeta) * P / nvd * (vd0 * vd0');
    end
    R = hermitian_project(R);
end

function R = design_eigen_opt(P, a0, ad0, v0, vd0, Q, b, L)
    N = length(v0);
    cvx_begin sdp quiet
        variable R(N,N) hermitian semidefinite
        variable t
        F = FIM_cvx_single(R, a0, ad0, v0, vd0, Q, b, L);
        maximize(t)
        subject to
            trace(R) == P;
            F - t * eye(3) >= 0;
    cvx_end
    R = hermitian_project(R);
end

function R = design_trace_opt(P, a0, ad0, v0, vd0, Q, b, L)
    N = length(v0);
    cvx_begin sdp quiet
        variable R(N,N) hermitian semidefinite
        variable u(3)
        F = FIM_cvx_single(R, a0, ad0, v0, vd0, Q, b, L);
        minimize(sum(u))
        subject to
            trace(R) == P;
            for k = 1:3
                ek = zeros(3,1);
                ek(k) = 1;
                [F, ek; ek', u(k)] >= 0;
            end
    cvx_end
    R = hermitian_project(R);
end

function R = design_det_opt(P, a0, ad0, v0, vd0, Q, b, L)
    A00 = real(a0' / Q * a0);
    A10 = ad0' / Q * a0;
    A11 = real(ad0' / Q * ad0);
    nv = real(v0' * v0);
    nvd = real(vd0' * vd0);
    alpha = L * abs(b)^2 * nv * real(A11 - abs(A10)^2 / A00);
    beta = L * abs(b)^2 * nvd * A00;
    if alpha >= beta/3
        lambda1 = P;
        lambda2 = 0;
    else
        lambda1 = 2 * beta * P / (3 * (beta - alpha));
        lambda2 = P - lambda1;
    end
    R = lambda1 / nv * (v0 * v0') + lambda2 / nvd * (vd0 * vd0');
    R = hermitian_project(R);
end

function F = FIM_numeric_single(R, a0, ad0, v0, vd0, Q, b, L)
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

function F = FIM_cvx_single(R, a0, ad0, v0, vd0, Q, b, L)
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

function R = hermitian_project(R)
    R = (R + R')/2;
end

function plot_all_methods(x, Y, methods, lineStyles, lineWidths, markerEvery)
    % Y is num_methods x num_points.
    hold on;
    for mm = 1:length(methods)
        y = Y(mm,:);

        switch methods{mm}
            case 'Eigen'
                plot(x, y, lineStyles{mm}, ...
                    'LineWidth', lineWidths(mm), ...
                    'MarkerIndices', 1:markerEvery:length(x), ...
                    'MarkerSize', 4);
            case 'Det'
                plot(x, y, lineStyles{mm}, ...
                    'LineWidth', lineWidths(mm), ...
                    'MarkerIndices', 1:markerEvery:length(x), ...
                    'MarkerSize', 4);
            otherwise
                plot(x, y, lineStyles{mm}, ...
                    'LineWidth', lineWidths(mm));
        end
    end

    legend({'Uncorrelated Waveforms', 'Sum Beam', 'Angle only', ...
            'Eigen Opt', 'Trace Opt', 'Det Opt'}, ...
            'Location', 'best');

    xlim([min(x), max(x)]);
end