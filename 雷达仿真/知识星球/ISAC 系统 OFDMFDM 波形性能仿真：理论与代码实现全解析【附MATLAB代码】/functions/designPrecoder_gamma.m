function W = designPrecoder_gamma(H, gamma)
% Single-stream precoder W (Nt x 1), gamma controls comm vs sensing weighting.

[~, Nt] = size(H);

% Communication-oriented: MRT toward the channel (use Frobenius matched direction)
W_comm = sum(conj(H), 1).';           % Nt x 1  (sum over Rx antennas)
W_comm = W_comm / norm(W_comm);

% Sensing-oriented: broadside steering (simple, can be replaced by a(theta))
W_sense = ones(Nt,1) / sqrt(Nt);

% Gamma-weighted spatial precoder
W = sqrt(1-gamma)*W_comm + sqrt(gamma)*W_sense;

% Normalize TX power
W = W / norm(W);
end
