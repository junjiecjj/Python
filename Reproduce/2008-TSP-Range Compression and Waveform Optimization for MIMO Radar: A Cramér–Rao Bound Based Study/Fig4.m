%% ============================================================
%  Reproduce Fig. 4 in:
%  Jian Li et al., "Range Compression and Waveform Optimization
%  for MIMO Radar: A Cramer-Rao Bound Based Study"
%
%  Fig. 4:
%    Root CRB versus initial angle estimate error
%    single-target case, theta = -16.5 deg, b = 1
%
%  Radar configurations:
%    A: MIMO Radar A(5, 0.5)
%    B: MIMO Radar B(0.5, 0.5)
%    C: MIMO Radar C(0.5, 5)
%
%  Waveforms:
%    1. Uncorrelated waveforms
%    2. Sum beam
%    3. Angle-only
%    4. Eigen-Opt
%    5. Trace-Opt
%    6. Det-Opt
%
%  Requirements:
%    CVX is needed for Eigen-Opt and Trace-Opt.
%    Det-Opt here uses 1-D search, not CVX log_det.
%% ============================================================

clear; clc; close all;

%% ========== Global parameters ==========
M = 10;                 % transmit antennas
N = 10;                 % receive antennas
P = 1;                  % total transmit power

theta_true = -16.5;     % true target angle, deg
theta_jam  = 5;         % jammer angle, deg
b_true = 1;             % target complex amplitude

sigma2 = 1;
AINR_dB = 100;
AINR = 10^(AINR_dB/10);

% Initial angle estimate error grid, in deg.
% Use 0.1 for faster testing, 0.05 or 0.025 for smoother curves.
err_grid = -3:0.1:3;

% Radar configurations: [dt, dr]
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
num_methods = length(methods);
num_radars = length(radars);
num_err = length(err_grid);

% Store results.
% Dimension:
%   radar index x method index x error index
RCRB_theta = zeros(num_radars, num_methods, num_err);
RCRB_b     = zeros(num_radars, num_methods, num_err);

%% ========== Main loop ==========
for rr = 1:num_radars

    dt = radars(rr).dt;
    dr = radars(rr).dr;

    fprintf('\n===== Radar %s =====\n', radars(rr).name);

    % Array positions.
    tx_pos = ((0:M-1) - (M-1)/2).' * dt;
    rx_pos = ((0:N-1) - (N-1)/2).' * dr;

    % Steering functions.
    vfun  = @(theta) exp(1j*2*pi*tx_pos*sind(theta));
    afun  = @(theta) exp(1j*2*pi*rx_pos*sind(theta));

    vdfun = @(theta) 1j*2*pi*tx_pos*cosd(theta)*(pi/180) ...
                     .* exp(1j*2*pi*tx_pos*sind(theta));

    adfun = @(theta) 1j*2*pi*rx_pos*cosd(theta)*(pi/180) ...
                     .* exp(1j*2*pi*rx_pos*sind(theta));

    % Jammer-plus-noise covariance Q.
    aj = afun(theta_jam);
    jammer_power = AINR * sigma2 / N;
    Q = sigma2 * eye(N) + jammer_power * (aj * aj');
    Qinv = inv(Q);

    % True target quantities for evaluating the real CRB.
    a_true  = afun(theta_true);
    v_true  = vfun(theta_true);
    ad_true = adfun(theta_true);
    vd_true = vdfun(theta_true);

    for ii = 1:num_err

        err = err_grid(ii);
        theta_hat = theta_true + err;

        fprintf('Radar %s, error %.2f deg, %d / %d\n', ...
            radars(rr).name, err, ii, num_err);

        % Design quantities based on initial estimate theta_hat.
        a_hat  = afun(theta_hat);
        v_hat  = vfun(theta_hat);
        ad_hat = adfun(theta_hat);
        vd_hat = vdfun(theta_hat);

        %% ----- 1. Uncorrelated waveforms -----
        R_uncorr = (P/M) * eye(M);

        %% ----- 2. Sum beam -----
        R_sum = P / real(v_hat' * v_hat) * (v_hat * v_hat');

        %% ----- 3. Angle-only -----
        R_angle = design_angle_only(P, v_hat, vd_hat, a_hat, ad_hat, Qinv, b_true);

        %% ----- 4. Eigen-Opt -----
        R_eigen = design_eigen_opt_subspace(P, v_hat, vd_hat, a_hat, ad_hat, Qinv, b_true);

        %% ----- 5. Trace-Opt -----
        R_trace = design_trace_opt_subspace(P, v_hat, vd_hat, a_hat, ad_hat, Qinv, b_true);

        %% ----- 6. Det-Opt -----
        R_det = design_det_opt_1d(P, v_hat, vd_hat, a_hat, ad_hat, Qinv, b_true);

        R_list = {R_uncorr, R_sum, R_angle, R_eigen, R_trace, R_det};

        %% ----- Evaluate true CRB for every designed waveform -----
        for mm = 1:num_methods
            R_now = hermitian_project(R_list{mm});

            F_true = FIM_numeric_single( ...
                R_now, a_true, ad_true, v_true, vd_true, Qinv, b_true);

            C_true = inv(F_true);

            % Root CRB of theta, unit is degree because derivatives are
            % taken with respect to theta in degrees.
            RCRB_theta(rr, mm, ii) = sqrt(max(real(C_true(1,1)), 0));

            % Root CRB of complex amplitude b.
            % b = b_R + j b_I, so use sqrt(C_bR + C_bI).
            RCRB_b(rr, mm, ii) = sqrt(max(real(C_true(2,2) + C_true(3,3)), 0));
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
plot_all_methods(err_grid, squeeze(RCRB_theta(1,:,:)), ...
    methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Angle (deg)');
title('(a) Root CRB of \theta for MIMO radar A(5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-3 1e0]);

subplot(3,2,2);
plot_all_methods(err_grid, squeeze(RCRB_b(1,:,:)), ...
    methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Amplitude');
title('(b) Root CRB of b for MIMO radar A(5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-2 1e1]);

% -------- Radar B --------
subplot(3,2,3);
plot_all_methods(err_grid, squeeze(RCRB_theta(2,:,:)), ...
    methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Angle (deg)');
title('(c) Root CRB of \theta for MIMO radar B(0.5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-3 1e0]);

subplot(3,2,4);
plot_all_methods(err_grid, squeeze(RCRB_b(2,:,:)), ...
    methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Amplitude');
title('(d) Root CRB of b for MIMO radar B(0.5,0.5)');
set(gca, 'YScale', 'log');
ylim([1e-2 1e1]);

% -------- Radar C --------
subplot(3,2,5);
plot_all_methods(err_grid, squeeze(RCRB_theta(3,:,:)), ...
    methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Angle (deg)');
title('(e) Root CRB of \theta for MIMO radar C(0.5,5)');
set(gca, 'YScale', 'log');
ylim([1e-3 1e0]);

subplot(3,2,6);
plot_all_methods(err_grid, squeeze(RCRB_b(3,:,:)), ...
    methods, lineStyles, lineWidths, markerEvery);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of Amplitude');
title('(f) Root CRB of b for MIMO radar C(0.5,5)');
set(gca, 'YScale', 'log');
ylim([1e-2 1e1]);

%% ============================================================
%  Local functions
%% ============================================================

function R = design_angle_only(P, v0, vd0, a0, ad0, Qinv, b)
    % Single-target Angle-only criterion.
    %
    % Paper Appendix D reduces it to:
    %   max alpha*lambda11 + beta*lambda22
    %   s.t. lambda11 + lambda22 = P
    %
    % If alpha > beta: put all power on v.
    % If alpha < beta: almost all power on vdot, with tiny zeta on v.
    % If alpha = beta: any convex combination works.

    A00 = real(a0'  * Qinv * a0);
    A10 = ad0' * Qinv * a0;
    A11 = real(ad0' * Qinv * ad0);

    nv  = real(v0'  * v0);
    nvd = real(vd0' * vd0);

    alpha = abs(b)^2 * nv  * real(A11 - abs(A10)^2 / A00);
    beta  = abs(b)^2 * nvd * A00;

    u1 = v0 / norm(v0);

    u2 = vd0 - u1 * (u1' * vd0);
    if norm(u2) < 1e-12
        u2 = null(u1');
        u2 = u2(:,1);
    else
        u2 = u2 / norm(u2);
    end

    if alpha > beta
        lambda1 = P;
        lambda2 = 0;
    elseif alpha < beta
        % For alpha < beta, the paper says the optimum is not attained
        % exactly if lambda11 must be positive. Use tiny zeta.
        zeta = 1e-4;
        lambda1 = zeta * P;
        lambda2 = (1-zeta) * P;
    else
        lambda1 = P/2;
        lambda2 = P/2;
    end

    R = lambda1 * (u1*u1') + lambda2 * (u2*u2');
    R = hermitian_project(R);
end


function R = design_eigen_opt_subspace(P, v0, vd0, a0, ad0, Qinv, b)
    % Eigen-Opt:
    %   maximize lambda_min(FIM)
    %
    % Use the optimal subspace span{v, vdot} to reduce dimension.

    U = make_two_dim_subspace(v0, vd0);

    cvx_begin sdp quiet
        variable X(2,2) hermitian semidefinite
        variable t

        Rcvx = U * X * U';
        F = FIM_cvx_single(Rcvx, a0, ad0, v0, vd0, Qinv, b);

        maximize(t)
        subject to
            trace(X) == P;
            F - t * eye(3) >= 0;
    cvx_end

    R = U * X * U';
    R = hermitian_project(R);
end


function R = design_trace_opt_subspace(P, v0, vd0, a0, ad0, Qinv, b)
    % Trace-Opt:
    %   minimize trace(inv(FIM))
    %
    % SDP epigraph form:
    %   [F e_k; e_k' u_k] >= 0

    U = make_two_dim_subspace(v0, vd0);

    cvx_begin sdp quiet
        variable X(2,2) hermitian semidefinite
        variable u(3)

        Rcvx = U * X * U';
        F = FIM_cvx_single(Rcvx, a0, ad0, v0, vd0, Qinv, b);

        minimize(sum(u))
        subject to
            trace(X) == P;

            for k = 1:3
                ek = zeros(3,1);
                ek(k) = 1;
                [F, ek; ek', u(k)] >= 0;
            end
    cvx_end

    R = U * X * U';
    R = hermitian_project(R);
end


function R = design_det_opt_1d(P, v0, vd0, a0, ad0, Qinv, b)
    % Det-Opt without CVX log_det.
    %
    % Search in:
    %   R(lambda) = lambda u1 u1^H + (P-lambda) u2 u2^H

    U = make_two_dim_subspace(v0, vd0);
    u1 = U(:,1);
    u2 = U(:,2);

    obj = @(lambda1) -logdet_numeric( ...
        FIM_numeric_single( ...
            lambda1 * (u1*u1') + (P-lambda1) * (u2*u2'), ...
            a0, ad0, v0, vd0, Qinv, b ...
        ) ...
    );

    lambda1_opt = fminbnd(obj, 1e-8, P-1e-8);
    lambda2_opt = P - lambda1_opt;

    R = lambda1_opt * (u1*u1') + lambda2_opt * (u2*u2');
    R = hermitian_project(R);
end


function U = make_two_dim_subspace(v0, vd0)
    % Orthonormal basis for span{v0, vd0}.

    u1 = v0 / norm(v0);

    u2 = vd0 - u1 * (u1' * vd0);

    if norm(u2) < 1e-12
        Z = null(u1');
        u2 = Z(:,1);
    else
        u2 = u2 / norm(u2);
    end

    U = [u1, u2];
end


function F = FIM_numeric_single(R, a0, ad0, v0, vd0, Qinv, b)
    % Numeric 3 x 3 real Fisher information matrix.
    %
    % Parameter vector:
    %   eta = [theta, real(b), imag(b)]^T
    %
    % theta derivative is with respect to degrees, so CRB(theta)
    % is in deg^2.

    R = hermitian_project(R);

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

    F = real((F + F')/2);
end


function F = FIM_cvx_single(R, a0, ad0, v0, vd0, Qinv, b)
    % CVX-compatible FIM.
    % All expressions are affine in R.

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


function y = logdet_numeric(F)
    % Stable numeric log-det.

    F = real((F + F')/2);
    ev = eig(F);
    ev = max(real(ev), 1e-12);
    y = sum(log(ev));
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