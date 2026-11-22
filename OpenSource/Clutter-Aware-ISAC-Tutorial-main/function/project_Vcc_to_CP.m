function Vcp = project_Vcc_to_CP(V, Nr, Nt, eps_load)
% Project the clutter kernel V onto the completely positive (CP) set
% so that the induced clutter covariance matrix Rcc remains PSD.
%
% Inputs:
%   V        : Clutter kernel of size (Nr^2) x (Nt^2)
%   Nr       : Number of receive antennas
%   Nt       : Number of transmit antennas
%   eps_load : Optional diagonal loading factor for numerical robustness
%
% Output:
%   Vcp      : CP-projected clutter kernel

% Reshape V into its 4-D tensor form
T = reshape(V, [Nr, Nr, Nt, Nt]);   % Indices: (a, b, i, j)

% Construct the Choi matrix associated with the clutter kernel
J = reshape(permute(T, [1 3 2 4]), [Nr * Nt, Nr * Nt]);
J = (J + J') / 2;   % Enforce Hermitian symmetry

% Project the Choi matrix onto the PSD cone
[U, D] = eig(J);
d = real(diag(D));
d(d < 0) = 0;

% Optional diagonal loading for numerical robustness
if nargin >= 4 && eps_load > 0
    scl = max(1, mean(d));
    d = d + eps_load * scl;
end
Jp = U * diag(d) * U';
Jp = (Jp + Jp') / 2;

% Map the projected Choi matrix back to the kernel form
Tp = ipermute(reshape(Jp, [Nr, Nt, Nr, Nt]), [1 3 2 4]);
Vcp = reshape(Tp, [Nr * Nr, Nt * Nt]);
end