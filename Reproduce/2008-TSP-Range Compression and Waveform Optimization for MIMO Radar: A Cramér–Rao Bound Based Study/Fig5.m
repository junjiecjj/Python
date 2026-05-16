%% ============================================================
%  Reproduce Fig. 5 in:
%  Jian Li et al., "Range Compression and Waveform Optimization
%  for MIMO Radar: A Cramer-Rao Bound Based Study"
%
%  Fig. 5:
%    Optimal transmit beampatterns for the two-target case
%    theta1 = -16.5 deg, theta2 = -10 deg
%    b1 = 1, b2 = 20
%
%    Trace-Opt criterion
%      (a) MIMO Radar A(5,0.5)
%      (b) MIMO Radar B(0.5,0.5)
%
%  Requirement:
%    CVX
%% ============================================================

clear; clc; close all;

%% ========== Global parameters ==========
M = 10;                 % transmit antennas
N = 10;                 % receive antennas
P = 1;                  % total transmit power

theta = [-16.5; -10];   % two target angles, degrees
b = [1; 20];            % two complex amplitudes

theta_jam = 5;          % jammer angle, degrees
sigma2 = 1;

AINR_dB = 100;
AINR = 10^(AINR_dB/10);

angle_grid = -25:0.02:5;

% Radar configurations for Fig. 5
radars(1).name = 'A(5,0.5)';
radars(1).dt = 5;
radars(1).dr = 0.5;

radars(2).name = 'B(0.5,0.5)';
radars(2).dt = 0.5;
radars(2).dr = 0.5;

R_opt_all = cell(2,1);
B_dB_all = cell(2,1);

%% ========== Main loop over two radars ==========
for rr = 1:2

    dt = radars(rr).dt;
    dr = radars(rr).dr;

    fprintf('\n===== Designing Trace-Opt for MIMO Radar %s =====\n', ...
        radars(rr).name);

    %% Array geometry
    tx_pos = ((0:M-1) - (M-1)/2).' * dt;
    rx_pos = ((0:N-1) - (N-1)/2).' * dr;

    vfun  = @(th) exp(1j*2*pi*tx_pos*sind(th));
    afun  = @(th) exp(1j*2*pi*rx_pos*sind(th));

    vdfun = @(th) 1j*2*pi*tx_pos*cosd(th)*(pi/180) ...
                  .* exp(1j*2*pi*tx_pos*sind(th));

    adfun = @(th) 1j*2*pi*rx_pos*cosd(th)*(pi/180) ...
                  .* exp(1j*2*pi*rx_pos*sind(th));

    %% Interference-plus-noise covariance Q
    aj = afun(theta_jam);
    jammer_power = AINR * sigma2 / N;
    Q = sigma2 * eye(N) + jammer_power * (aj * aj');
    Qinv = inv(Q);

    %% Steering matrices
    A  = zeros(N,2);
    Ad = zeros(N,2);
    V  = zeros(M,2);
    Vd = zeros(M,2);

    for k = 1:2
        A(:,k)  = afun(theta(k));
        Ad(:,k) = adfun(theta(k));
        V(:,k)  = vfun(theta(k));
        Vd(:,k) = vdfun(theta(k));
    end

    %% Trace-Opt for CRB block of target 1 only
    R_opt = design_trace_opt_two_target_target1( ...
        P, A, Ad, V, Vd, Qinv, b);

    R_opt_all{rr} = R_opt;

    %% Beampattern
    B_dB_all{rr} = tx_beampattern(R_opt, vfun, angle_grid);

end

%% ========== Plot Fig. 5 style ==========
figure('Color','w', 'Position', [120 120 900 360]);

subplot(1,2,1);
plot(angle_grid, B_dB_all{1}, 'LineWidth', 1.4);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(a) MIMO radar A(5,0.5)');
xlim([-25 5]);
ylim([-30 0]);
xline(theta(1), 'k--', 'LineWidth', 1.0);
xline(theta(2), 'k:',  'LineWidth', 1.0);
legend('Trace-Opt', '\theta_1', '\theta_2', 'Location', 'best');

subplot(1,2,2);
plot(angle_grid, B_dB_all{2}, 'LineWidth', 1.4);
grid on;
xlabel('Angle (deg)');
ylabel('Beampattern (dB)');
title('(b) MIMO radar B(0.5,0.5)');
xlim([-25 5]);
ylim([-30 0]);
xline(theta(1), 'k--', 'LineWidth', 1.0);
xline(theta(2), 'k:',  'LineWidth', 1.0);
legend('Trace-Opt', '\theta_1', '\theta_2', 'Location', 'best');

%% ============================================================
%  Local functions
%% ============================================================

function R = design_trace_opt_two_target_target1(P, A, Ad, V, Vd, Qinv, b)
    % Trace-Opt for two-target case.
    %
    % Unknown parameter vector:
    %   eta = [theta1, theta2, real(b1), real(b2), imag(b1), imag(b2)]^T
    %
    % FIM is 6 x 6.
    %
    % Fig. 5 uses the generalized Trace-Opt criterion, minimizing
    % the CRB block of target 1 only:
    %
    %   target 1 parameters are:
    %       theta1, real(b1), imag(b1)
    %
    % In the above ordering, their indices are:
    %       [1, 3, 5]
    %
    % Therefore use Schur/LMI form for those three diagonal entries:
    %
    %   minimize sum u_k
    %   subject to [F e_i; e_i' u_k] >= 0

    M = size(V,1);

    cvx_begin sdp quiet
        variable R(M,M) hermitian semidefinite
        variable u(3)

        F = FIM_cvx_multi(R, A, Ad, V, Vd, Qinv, b);

        minimize(sum(u))
        subject to
            trace(R) == P;

            idx = [1, 3, 5];   % theta1, real(b1), imag(b1)

            for kk = 1:3
                e = zeros(6,1);
                e(idx(kk)) = 1;
                [F, e; e', u(kk)] >= 0;
            end
    cvx_end

    R = hermitian_project(R);
end


function F = FIM_cvx_multi(R, A, Ad, V, Vd, Qinv, b)
    % CVX-compatible FIM for multiple targets.
    %
    % Model:
    %   X = sum_k b_k a_k v_k^T S + noise
    %
    % Parameter vector:
    %   eta = [theta_1 ... theta_K,
    %          real(b_1) ... real(b_K),
    %          imag(b_1) ... imag(b_K)]^T
    %
    % F_ij = 2 Re{ trace( dM_i^H Q^{-1} dM_j ) }

    K = length(b);

    Ftt = cvx(zeros(K,K));
    Ftr = cvx(zeros(K,K));
    Fti = cvx(zeros(K,K));
    Frr = cvx(zeros(K,K));
    Fri = cvx(zeros(K,K));
    Fii = cvx(zeros(K,K));

    for p = 1:K
        for q = 1:K

            % Receive inner products
            A00 = A(:,p)'  * Qinv * A(:,q);
            Ad0 = Ad(:,p)' * Qinv * A(:,q);
            A0d = A(:,p)'  * Qinv * Ad(:,q);
            Add = Ad(:,p)' * Qinv * Ad(:,q);

            % Transmit covariance inner products
            V00 = V(:,p)'  * R * V(:,q);
            Vd0 = Vd(:,p)' * R * V(:,q);
            V0d = V(:,p)'  * R * Vd(:,q);
            Vdd = Vd(:,p)' * R * Vd(:,q);

            % theta-theta block
            Gtt = conj(b(p)) * b(q) * ...
                  (Add * V00 + Ad0 * V0d + A0d * Vd0 + A00 * Vdd);

            % theta-real(b) block
            Gtr = conj(b(p)) * ...
                  (Ad0 * V00 + A00 * Vd0);

            % theta-imag(b) block
            % derivative wrt imag(b_q) is j * a_q v_q^T S
            Gti = 1j * Gtr;

            % real-real amplitude block
            Grr = A00 * V00;

            % real-imag block
            Gri = 1j * Grr;

            % imag-imag block
            % derivative wrt imag(b) has j factor on both sides
            Gii = Grr;

            Ftt(p,q) = 2 * real(Gtt);
            Ftr(p,q) = 2 * real(Gtr);
            Fti(p,q) = 2 * real(Gti);

            Frr(p,q) = 2 * real(Grr);
            Fri(p,q) = 2 * real(Gri);
            Fii(p,q) = 2 * real(Gii);
        end
    end

    F = [Ftt, Ftr, Fti;
         Ftr.', Frr, Fri;
         Fti.', Fri.', Fii];

    F = 0.5 * (F + F.');
end


function B_dB = tx_beampattern(R, vfun, angle_grid)
    % Transmit beampattern:
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


function R = hermitian_project(R)
    R = (R + R')/2;
end