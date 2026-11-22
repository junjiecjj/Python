function SINR_matrix = comp_SINR_comm(Rxx_n, R_nk, H_comm, sigma2)
% Compute the communication SINR for all users and subcarriers.
%
% Inputs:
%   Rxx_n    : Total transmit covariance over subcarriers
%   R_nk     : User-specific covariance matrices
%   H_comm   : Communication channel tensor
%   sigma2   : Communication noise power
%
% Output:
%   SINR_matrix : SINR values of size N x Ku

[~, ~, N, Ku] = size(R_nk);
SINR_matrix = zeros(N, Ku);
for n = 1:N
    for k = 1:Ku
        h_nk = H_comm(:, n, k);
        H_nk = h_nk * h_nk';
        signal_power = real(trace(H_nk * R_nk(:, :, n, k)));
        interference_power = real(trace(H_nk * Rxx_n(:, :, n))) - signal_power;
        SINR_matrix(n, k) = signal_power / (interference_power + sigma2);
    end
end
end