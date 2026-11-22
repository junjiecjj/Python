function Ac = build_Ac(Vcc_n, u_n, Nt)
% Construct the clutter-dependent matrix Ac for a fixed receive beamformer.
%
% Inputs:
%   Vcc_n : Clutter kernel at the n-th subcarrier
%   u_n   : Receive beamformer at the n-th subcarrier
%   Nt    : Number of transmit antennas
%
% Output:
%   Ac    : Clutter-related matrix for transmit covariance optimization

uu = u_n * u_n';
z = Vcc_n' * uu(:);
Ac = reshape(z, Nt, Nt);
Ac = (Ac + Ac') / 2;
end