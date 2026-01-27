function SINR_i = compute_user_SINR(H, T, N0)
% COMPUTE_USER_SINR  Compute the instantaneous SINR for each user.
%
%   SINR_i = compute_user_SINR(H, T, N0) returns a K×1 vector of
%   instantaneous SINR values (linear scale) for a multi‑user MISO
%   downlink with channel matrix H and beamforming matrix T.  The
%   received signal for user i is y_i = h_i^T sum_k t_k d_k + n_i,
%   where t_k is the k‑th column of T.  The SINR is defined as
%       |h_i^T t_i|^2 / (sum_{k≠i} |h_i^T t_k|^2 + N0).
%
%   Inputs:
%     H  – N×K channel matrix (columns h_i)
%     T  – N×K beamforming matrix (columns t_i)
%     N0 – noise variance at each user
%
%   Output:
%     SINR_i – K×1 vector of SINR values (linear scale)

[~, K] = size(H);
SINR_i = zeros(K,1);
for i = 1:K
    hi  = H(:,i);
    hiT = hi.' * T;   % row vector of dimension 1×K
    signal = abs(hiT(i))^2;
    interference = sum(abs(hiT).^2) - signal;
    SINR_i(i) = signal / (interference + N0);
end
end