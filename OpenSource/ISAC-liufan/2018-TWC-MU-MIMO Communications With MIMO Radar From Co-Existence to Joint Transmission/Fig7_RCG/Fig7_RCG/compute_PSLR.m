function [PSLR_dB, P_theta] = compute_PSLR(T, theta_grid, steer_fun, theta0, theta1, theta2)
% COMPUTE_PSLR  Compute the peak sidelobe ratio of a transmit beampattern.
%
%   [PSLR_dB, P_theta] = compute_PSLR(T, theta_grid, steer_fun, theta0, theta1, theta2)
%
%   Given a beamforming matrix T (N×K), the covariance C = T T^H is
%   used to compute the power beampattern P(theta) = a(theta)^H C a(theta)
%   on a grid of angles.  The peak sidelobe ratio (PSLR) is then
%   calculated as the ratio of the peak value within the main lobe
%   to the peak value outside the main lobe.  Angles theta1 and theta2
%   define the main lobe region around the boresight theta0.
%
%   Inputs:
%     T         – N×K beamforming matrix
%     theta_grid – vector of angles at which to evaluate the beampattern
%     steer_fun  – handle returning the steering vector for a given angle
%     theta0, theta1, theta2 – main lobe center and edges (degrees)
%
%   Outputs:
%     PSLR_dB – peak sidelobe ratio in decibels
%     P_theta – beampattern values on the grid (linear scale)

C = T * T';
nTheta = numel(theta_grid);
P_theta = zeros(nTheta,1);
for m = 1:nTheta
    a = steer_fun(theta_grid(m)).';
    P_theta(m) = real(a' * C * a);
end

% Indices for main lobe region
main_idx = find(theta_grid >= theta1 & theta_grid <= theta2);
side_idx = setdiff(1:nTheta, main_idx);

P_main = max(P_theta(main_idx));
P_side = max(P_theta(side_idx));

PSLR_dB = 10 * log10(P_main / P_side);
end