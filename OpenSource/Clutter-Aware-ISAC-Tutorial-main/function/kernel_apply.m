function Rcc = kernel_apply(Vcc_n, RX_n, Nr)
% Apply the clutter kernel to a transmit covariance matrix.
%
% Inputs:
%   Vcc_n : Clutter kernel at the n-th subcarrier
%   RX_n  : Transmit covariance matrix at the n-th subcarrier
%   Nr    : Number of receive antennas
%
% Output:
%   Rcc   : Induced clutter covariance matrix

v = Vcc_n * RX_n(:);
Rcc = reshape(v, Nr, Nr);
Rcc = (Rcc + Rcc') / 2;
end