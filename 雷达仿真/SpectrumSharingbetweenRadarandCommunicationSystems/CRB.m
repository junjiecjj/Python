function [ crb ] = CRB( Rs, a_t, a_tdiff, a_rdiff, N_R, snr)
% This function calculates Cramer Rao Bound
% Inputs:
%   Rs: Radar signal covariance matrix
%   a_t: Transmit steering vector
%   a_tdiff: Derivative of transmit steering vector
%   a_rdiff: Derivative of receive steering vector
%   N_R: Number of receive antenna elements
%   snr: Signal-to-noise ratio (linear scale)
% Output:
%   crb: Cramer-Rao Bound value

T1 = N_R * a_tdiff' * Rs' * a_tdiff;
T2 = a_t' * Rs' * a_t * ((norm(a_rdiff, 2))^2);
T3 = N_R * abs(a_t' * Rs' * a_tdiff)^2;
T4 = a_t' * Rs' * a_t;
T = T1 + T2 - (T3 ./ T4);
crb = 1 / (2 * snr * T);
end