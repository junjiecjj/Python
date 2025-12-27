function R = solve_radar_3dB_cvx(N, P0, d, theta0, theta1, theta2, theta_sidelobe)
% SOLVE_RADAR_3DB_CVX  Design a covariance matrix with a 3 dB main lobe.
%
%   R = solve_radar_3dB_cvx(N, P0, d, theta0, theta1, theta2, theta_sidelobe)
%
%   This function formulates and solves the convex optimization problem
%   described in Eq. (10) of the paper to design a radar transmit
%   covariance matrix R.  The main lobe is centered at angle theta0 and
%   has 3 dB points at theta1 and theta2.  The sidelobe region is defined
%   by theta_sidelobe (a vector of angles).  The diagonal elements of R
%   are fixed such that the total transmit power is P0.
%
%   Inputs:
%     N             – number of transmit antennas
%     P0            – total transmit power (linear scale)
%     d             – inter‑element spacing (normalized by wavelength)
%     theta0        – center of main beam (degrees)
%     theta1, theta2– 3 dB points of main beam (degrees)
%     theta_sidelobe– vector of angles describing sidelobe region
%
%   Output:
%     R             – N×N covariance matrix satisfying the design
%
%   NOTE:  This code requires CVX (a package for specifying and
%   solving convex programs).  If CVX is not installed, download it
%   from http://cvxr.com/cvx/ and run cvx_setup.

deg2rad = pi/180;

% Steering vectors at key angles
a0 = exp(1j*2*pi*d*sin(theta0*deg2rad)*(0:N-1).').';
a1 = exp(1j*2*pi*d*sin(theta1*deg2rad)*(0:N-1).').';
a2 = exp(1j*2*pi*d*sin(theta2*deg2rad)*(0:N-1).').';

% Sidelobe steering matrix
Msl = numel(theta_sidelobe);
Asl = zeros(N, Msl);
for m = 1:Msl
    th = theta_sidelobe(m);
    Asl(:,m) = exp(1j*2*pi*d*sin(th*deg2rad)*(0:N-1).').';
end

% Solve the SDP.  We maximize t such that the main beam power exceeds
% the sidelobe power by at least t and the 3 dB constraints hold.
cvx_begin sdp quiet
    variable R(N,N) complex semidefinite
    variable t

    % Main‑beam power at theta0
    P0_main = a0 * R * a0';

    % 3 dB constraints at theta1 and theta2
    P1 = a1 * R * a1';
    P2 = a2 * R * a2';

    maximize( t )

    subject to
        % Sidelobe inequality: main beam exceeds sidelobes by >= t
        for m = 1:Msl
            am = Asl(:,m).';
            Pm = am * R * am';
            P0_main - Pm >= t;
        end

        % 3 dB constraints (equalities)
        P1 == P0_main/2;
        P2 == P0_main/2;

        % Per‑antenna power constraint (diagonal)
        diag(R) == (P0/N) * ones(N,1);

        % Hermitian symmetry
        R == R';
cvx_end

end