function scnr = compute_SCNR(RX, u, Vcc, R_eta, sigma2_tar, a_t, b_t)
% Compute the overall SCNR across all subcarriers.
%
% Inputs:
%   RX         : Transmit covariance matrices
%   u          : Receive beamformers
%   Vcc        : Clutter kernel over subcarriers
%   R_eta      : Interference-plus-noise covariance over subcarriers
%   sigma2_tar : Target power gain over subcarriers
%   a_t, b_t   : Target steering vectors
%
% Output:
%   scnr       : Overall SCNR

N = numel(Vcc);
Nr = size(u, 1);
num = 0;
den = 0;
for n = 1:N
    g = real(a_t{n}' * RX(:, :, n) * a_t{n});
    h = abs(u(:, n)' * b_t{n})^2;
    num = num + sigma2_tar(n) * h * g;
    Rcc = kernel_apply(Vcc{n}, RX(:, :, n), Nr);
    den = den + real(u(:, n)' * (Rcc + R_eta(:, :, n)) * u(:, n));
end
scnr = num / den;
end