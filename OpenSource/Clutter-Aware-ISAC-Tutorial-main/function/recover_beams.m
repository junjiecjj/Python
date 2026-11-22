function [W_comm, W_sens] = recover_beams(RX, Rnk, h_comm, Ns)
% Recover communication and sensing beams from covariance solutions.
%
% Inputs:
%   RX      : Total transmit covariance over subcarriers
%   Rnk     : User-specific covariance matrices
%   h_comm  : Communication channel tensor
%   Ns      : Total number of transmit streams
%
% Outputs:
%   W_comm  : Recovered communication beams
%   W_sens  : Recovered sensing beams

[Nt, ~, N, K] = size(Rnk);
S = Ns - K;
W_comm = zeros(Nt, K, N);
W_sens = zeros(Nt, S, N);
for n = 1:N
    % Recover communication beams
    for k = 1:K
        hk = h_comm(:, n, k);
        Rnkn = Rnk(:, :, n, k);
        denom = real(hk' * Rnkn * hk);
        if denom <= 0
            denom = 1e-12;
        end
        W_comm(:, k, n) = (1 / sqrt(denom)) * (Rnkn * hk);
    end

    % Residual covariance for sensing beams
    Rres = RX(:, :, n);
    for k = 1:K
        wk = W_comm(:, k, n);
        Rres = Rres - wk * wk';
    end
    Rres = (Rres + Rres') / 2;

    % Extract sensing beams from the dominant eigenmodes
    [U, D] = eig(Rres);
    d = real(diag(D));
    [d, idx] = sort(d, 'descend');
    U = U(:, idx);
    Usel = U(:, 1:S);
    dsel = d(1:S);
    amp = sqrt(max(dsel, 0));
    W_sens(:, :, n) = Usel * diag(amp);
end
end