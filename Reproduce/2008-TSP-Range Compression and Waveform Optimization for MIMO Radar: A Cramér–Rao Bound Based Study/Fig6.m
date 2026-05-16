%% ============================================================
%  Reproduce Fig. 6 in:
%  Jian Li et al., "Range Compression and Waveform Optimization
%  for MIMO Radar: A Cramer-Rao Bound Based Study"
%
%  Corrected version:
%    1) Uncorrelated waveforms are fixed baselines.
%    2) Sum beam is a fixed baseline pointing to true/nominal theta1.
%    3) Only Trace-Opt is redesigned with theta1_hat = theta1 + error.
%    4) Multi-target FIM uses transmit inner product v_q^T R v_p^*
%
%  Two-target case:
%    theta1 = -16.5 deg
%    theta2 = -10 deg
%    b1 = 1
%    b2 = 20
%
%  Waveforms:
%    1. Uncorrelated waveforms
%    2. Fixed Sum beam toward theta1
%    3. Trace-Opt, minimizing CRB block of target 1 only
%
%  Requirement:
%    CVX
%% ============================================================

clear; clc; close all;

%% ========== Global parameters ==========
M = 10;                         % transmit antennas
N = 10;                         % receive antennas
P = 1;                          % total transmit power

theta_true = [-16.5; -10];      % true target angles, degrees
b_true = [1; 20];               % target amplitudes

theta_jam = 5;                  % jammer angle, degrees
sigma2 = 1;

AINR_dB = 100;
AINR = 10^(AINR_dB/10);

% Error of initial estimate of theta1 only.
% For fast test, use -3:0.5:3 first.
err_grid = -3:0.1:3;

% Radar configurations for Fig. 6.
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
RCRB_b1     = zeros(num_radars, num_methods, num_err);

%% ========== Main loop ==========
for rr = 1:num_radars

    dt = radars(rr).dt;
    dr = radars(rr).dr;

    fprintf('\n===== Radar %s =====\n', radars(rr).name);

    %% Array geometry
    tx_pos = ((0:M-1) - (M-1)/2).' * dt;
    rx_pos = ((0:N-1) - (N-1)/2).' * dr;

    vfun  = @(th) exp(1j*2*pi*tx_pos*sind(th));
    afun  = @(th) exp(1j*2*pi*rx_pos*sind(th));

    % Derivatives with respect to angle in degrees.
    vdfun = @(th) 1j*2*pi*tx_pos*cosd(th)*(pi/180) ...
                  .* exp(1j*2*pi*tx_pos*sind(th));

    adfun = @(th) 1j*2*pi*rx_pos*cosd(th)*(pi/180) ...
                  .* exp(1j*2*pi*rx_pos*sind(th));

    %% Interference-plus-noise covariance Q
    aj = afun(theta_jam);

    % AINR = incident jammer power * N / sigma2
    jammer_power = AINR * sigma2 / N;

    Q = sigma2 * eye(N) + jammer_power * (aj * aj');
    Qinv = inv(Q);

    %% True steering matrices for evaluating true CRB
    [A_true, Ad_true, V_true, Vd_true] = build_steering_multi( ...
        theta_true, afun, adfun, vfun, vdfun);

    %% ========== Fixed baselines ==========
    % 1. Uncorrelated waveform baseline
    R_uncorr_fixed = (P/M) * eye(M);

    % 2. Fixed Sum beam baseline
    %
    % Important:
    % In Fig. 6, Sum beam is a fixed baseline, not redesigned with
    % theta1_hat. It points to the true/nominal target-of-interest angle.
    %
    % Because the model is v^T S, the phased-array transmit weight is
    % conj(v(theta1)), so:
    %
    %   R_sum = P / ||v||^2 * conj(v) * v^T
    %
    v1_sum = V_true(:,1);
    R_sum_fixed = P / real(v1_sum' * v1_sum) * (conj(v1_sum) * v1_sum.');
    R_sum_fixed = hermitian_project(R_sum_fixed);

    %% Evaluate fixed baselines once, then copy over all error points
    F_uncorr_true = FIM_numeric_multi_correct( ...
        R_uncorr_fixed, A_true, Ad_true, V_true, Vd_true, Qinv, b_true);

    C_uncorr_true = inv(F_uncorr_true);

    theta_uncorr_rcrb = sqrt(max(real(C_uncorr_true(1,1)), 0));
    b_uncorr_rcrb     = sqrt(max(real(C_uncorr_true(3,3) + C_uncorr_true(5,5)), 0));

    F_sum_true = FIM_numeric_multi_correct( ...
        R_sum_fixed, A_true, Ad_true, V_true, Vd_true, Qinv, b_true);

    C_sum_true = inv(F_sum_true);

    theta_sum_rcrb = sqrt(max(real(C_sum_true(1,1)), 0));
    b_sum_rcrb     = sqrt(max(real(C_sum_true(3,3) + C_sum_true(5,5)), 0));

    RCRB_theta1(rr,1,:) = theta_uncorr_rcrb;
    RCRB_b1(rr,1,:)     = b_uncorr_rcrb;

    RCRB_theta1(rr,2,:) = theta_sum_rcrb;
    RCRB_b1(rr,2,:)     = b_sum_rcrb;

    %% ========== Trace-Opt loop over initial angle errors ==========
    for ii = 1:num_err

        err = err_grid(ii);

        fprintf('Radar %s, error %.2f deg, %d / %d\n', ...
            radars(rr).name, err, ii, num_err);

        %% Initial parameter estimates
        % Only theta1 has initial estimation error.
        theta_hat = theta_true;
        theta_hat(1) = theta_true(1) + err;

        % Other parameters are assumed exact.
        b_hat = b_true;

        [A_hat, Ad_hat, V_hat, Vd_hat] = build_steering_multi( ...
            theta_hat, afun, adfun, vfun, vdfun);

        %% Trace-Opt for target 1 CRB block
        R_trace = design_trace_opt_two_target_target1( ...
            P, A_hat, Ad_hat, V_hat, Vd_hat, Qinv, b_hat);

        %% Evaluate true CRB for Trace-Opt waveform
        F_trace_true = FIM_numeric_multi_correct( ...
            R_trace, A_true, Ad_true, V_true, Vd_true, Qinv, b_true);

        C_trace_true = inv(F_trace_true);

        % Parameter ordering:
        % eta = [theta1, theta2, real(b1), real(b2), imag(b1), imag(b2)]^T
        %
        % theta1 index    = 1
        % real(b1) index  = 3
        % imag(b1) index  = 5

        RCRB_theta1(rr,3,ii) = sqrt(max(real(C_trace_true(1,1)), 0));

        % Root CRB of complex amplitude b1.
        RCRB_b1(rr,3,ii) = sqrt(max(real(C_trace_true(3,3) + C_trace_true(5,5)), 0));
    end
end

%% ========== Plot Fig. 6 style ==========
figure('Color','w', 'Position', [100 100 980 680]);

lineSpec = {'k:', 'k-.', 'k-'};
lineWidth = [1.4, 1.4, 1.8];

%% -------- Radar A: theta1 --------
subplot(2,2,1);
plot_three_methods(err_grid, squeeze(RCRB_theta1(1,:,:)), ...
    lineSpec, lineWidth);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of \theta_1 (deg)');
title('(a) Root CRB of \theta_1 for MIMO radar A(5,0.5)');
set(gca, 'YScale', 'log');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

%% -------- Radar A: b1 --------
subplot(2,2,2);
plot_three_methods(err_grid, squeeze(RCRB_b1(1,:,:)), ...
    lineSpec, lineWidth);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of b_1');
title('(b) Root CRB of b_1 for MIMO radar A(5,0.5)');
set(gca, 'YScale', 'log');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

%% -------- Radar B: theta1 --------
subplot(2,2,3);
plot_three_methods(err_grid, squeeze(RCRB_theta1(2,:,:)), ...
    lineSpec, lineWidth);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of \theta_1 (deg)');
title('(c) Root CRB of \theta_1 for MIMO radar B(0.5,0.5)');
set(gca, 'YScale', 'log');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

%% -------- Radar B: b1 --------
subplot(2,2,4);
plot_three_methods(err_grid, squeeze(RCRB_b1(2,:,:)), ...
    lineSpec, lineWidth);
grid on;
xlabel('Error of Initial Angle Estimate (deg)');
ylabel('Root CRB of b_1');
title('(d) Root CRB of b_1 for MIMO radar B(0.5,0.5)');
set(gca, 'YScale', 'log');
xlim([min(err_grid), max(err_grid)]);
legend(methods, 'Location', 'best');

%% Optional: inspect numerical ranges
fprintf('\n===== Numerical ranges =====\n');
for rr = 1:num_radars
    fprintf('\nRadar %s\n', radars(rr).name);
    for mm = 1:num_methods
        fprintf('%s: theta1 RCRB [%.3e, %.3e], b1 RCRB [%.3e, %.3e]\n', ...
            methods{mm}, ...
            min(squeeze(RCRB_theta1(rr,mm,:))), max(squeeze(RCRB_theta1(rr,mm,:))), ...
            min(squeeze(RCRB_b1(rr,mm,:))),     max(squeeze(RCRB_b1(rr,mm,:))));
    end
end

%% ============================================================
%  Local functions
%% ============================================================

function [A, Ad, V, Vd] = build_steering_multi(theta, afun, adfun, vfun, vdfun)
    K = length(theta);

    N = length(afun(theta(1)));
    M = length(vfun(theta(1)));

    A  = zeros(N,K);
    Ad = zeros(N,K);
    V  = zeros(M,K);
    Vd = zeros(M,K);

    for k = 1:K
        A(:,k)  = afun(theta(k));
        Ad(:,k) = adfun(theta(k));
        V(:,k)  = vfun(theta(k));
        Vd(:,k) = vdfun(theta(k));
    end
end


function R = design_trace_opt_two_target_target1(P, A, Ad, V, Vd, Qinv, b)
    % Trace-Opt for target 1 only.
    %
    % Parameter ordering:
    % eta = [theta1, theta2, real(b1), real(b2), imag(b1), imag(b2)]^T
    %
    % Target 1 CRB block uses indices:
    %   theta1    -> 1
    %   real(b1)  -> 3
    %   imag(b1)  -> 5

    M = size(V,1);
    K = length(b);
    dim = 3*K;

    idx_target1 = [1, K+1, 2*K+1];

    cvx_begin sdp quiet
        variable R(M,M) hermitian semidefinite
        variable u(3)

        F = FIM_cvx_multi_correct(R, A, Ad, V, Vd, Qinv, b);

        minimize(sum(u))
        subject to
            trace(R) == P;

            for kk = 1:3
                e = zeros(dim,1);
                e(idx_target1(kk)) = 1;

                % Schur complement:
                % u(kk) >= e' * inv(F) * e
                [F, e; e', u(kk)] >= 0;
            end
    cvx_end

    R = hermitian_project(R);
end


function F = FIM_numeric_multi_correct(R, A, Ad, V, Vd, Qinv, b)
    % Correct numeric Fisher information matrix for K targets.
    %
    % Model:
    %   X = sum_k b_k a_k v_k^T S + noise
    %
    % Parameter ordering:
    %   eta = [theta_1 ... theta_K,
    %          real(b_1) ... real(b_K),
    %          imag(b_1) ... imag(b_K)]^T
    %
    % Because the signal uses v^T S, not v^H S,
    % the transmit-side inner product is:
    %
    %   y_q.' * R * conj(y_p)
    %
    % not:
    %
    %   y_p' * R * y_q

    K = length(b);
    R = hermitian_project(R);

    Ftt = zeros(K,K);
    Ftr = zeros(K,K);
    Fti = zeros(K,K);
    Frr = zeros(K,K);
    Fri = zeros(K,K);
    Fii = zeros(K,K);

    for p = 1:K
        for q = 1:K

            %% Receive-side inner products: x_p^H Q^{-1} x_q
            A00 = A(:,p)'  * Qinv * A(:,q);
            Ad0 = Ad(:,p)' * Qinv * A(:,q);
            A0d = A(:,p)'  * Qinv * Ad(:,q);
            Add = Ad(:,p)' * Qinv * Ad(:,q);

            %% Correct transmit-side inner products: y_q^T R y_p^*
            T_v_v   = V(:,q).'  * R * conj(V(:,p));
            T_v_vd  = Vd(:,q).' * R * conj(V(:,p));
            T_vd_v  = V(:,q).'  * R * conj(Vd(:,p));
            T_vd_vd = Vd(:,q).' * R * conj(Vd(:,p));

            %% theta_p versus theta_q
            Gtt = conj(b(p)) * b(q) * ...
                ( Add * T_v_v ...
                + Ad0 * T_v_vd ...
                + A0d * T_vd_v ...
                + A00 * T_vd_vd );

            %% theta_p versus real(b_q)
            Gtr = conj(b(p)) * ...
                ( Ad0 * T_v_v ...
                + A00 * T_vd_v );

            %% theta_p versus imag(b_q)
            % derivative wrt imag(b_q) is j*a_q*v_q^T*S
            Gti = 1j * Gtr;

            %% Amplitude blocks
            Grr = A00 * T_v_v;
            Gri = 1j * Grr;
            Gii = Grr;

            Ftt(p,q) = 2 * real(Gtt);
            Ftr(p,q) = 2 * real(Gtr);
            Fti(p,q) = 2 * real(Gti);

            Frr(p,q) = 2 * real(Grr);
            Fri(p,q) = 2 * real(Gri);
            Fii(p,q) = 2 * real(Gii);
        end
    end

    F = [Ftt,   Ftr,   Fti;
         Ftr.', Frr,   Fri;
         Fti.', Fri.', Fii];

    F = real((F + F')/2);
end


function F = FIM_cvx_multi_correct(R, A, Ad, V, Vd, Qinv, b)
    % Correct CVX-compatible Fisher information matrix for K targets.
    %
    % Parameter ordering:
    %   eta = [theta_1 ... theta_K,
    %          real(b_1) ... real(b_K),
    %          imag(b_1) ... imag(b_K)]^T

    K = length(b);

    Ftt = cvx(zeros(K,K));
    Ftr = cvx(zeros(K,K));
    Fti = cvx(zeros(K,K));
    Frr = cvx(zeros(K,K));
    Fri = cvx(zeros(K,K));
    Fii = cvx(zeros(K,K));

    for p = 1:K
        for q = 1:K

            %% Receive-side inner products
            A00 = A(:,p)'  * Qinv * A(:,q);
            Ad0 = Ad(:,p)' * Qinv * A(:,q);
            A0d = A(:,p)'  * Qinv * Ad(:,q);
            Add = Ad(:,p)' * Qinv * Ad(:,q);

            %% Correct transmit-side inner products
            T_v_v   = V(:,q).'  * R * conj(V(:,p));
            T_v_vd  = Vd(:,q).' * R * conj(V(:,p));
            T_vd_v  = V(:,q).'  * R * conj(Vd(:,p));
            T_vd_vd = Vd(:,q).' * R * conj(Vd(:,p));

            %% theta-theta block
            Gtt = conj(b(p)) * b(q) * ...
                ( Add * T_v_v ...
                + Ad0 * T_v_vd ...
                + A0d * T_vd_v ...
                + A00 * T_vd_vd );

            %% theta-real amplitude block
            Gtr = conj(b(p)) * ...
                ( Ad0 * T_v_v ...
                + A00 * T_vd_v );

            %% theta-imag amplitude block
            Gti = 1j * Gtr;

            %% amplitude-amplitude blocks
            Grr = A00 * T_v_v;
            Gri = 1j * Grr;
            Gii = Grr;

            Ftt(p,q) = 2 * real(Gtt);
            Ftr(p,q) = 2 * real(Gtr);
            Fti(p,q) = 2 * real(Gti);

            Frr(p,q) = 2 * real(Grr);
            Fri(p,q) = 2 * real(Gri);
            Fii(p,q) = 2 * real(Gii);
        end
    end

    F = [Ftt,   Ftr,   Fti;
         Ftr.', Frr,   Fri;
         Fti.', Fri.', Fii];

    F = 0.5 * (F + F.');
end


function R = hermitian_project(R)
    R = (R + R')/2;
end


function plot_three_methods(x, Y, lineSpec, lineWidth)
    % Y: num_methods x num_points

    hold on;

    for mm = 1:size(Y,1)
        plot(x, Y(mm,:), lineSpec{mm}, 'LineWidth', lineWidth(mm));
    end
end