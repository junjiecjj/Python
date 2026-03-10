function y = helperDecodeMIMOPMCW(xin, H)
% Combine received signals in the columns of xin based on the Hadamard
% matrix outer code in H
    [N, Nrx, Nb] = size(xin);
    Ntx = size(H, 1);
    L = N/Ntx;
    x = reshape(xin, L, Ntx, Nrx, Nb);
    y = zeros(L, Ntx, Nrx, Nb);

    for ntx = 1:Ntx
        h = repmat(H(ntx, :), L, 1, Nrx, Nb);
        y(:, ntx, :, :) = sum(x .* h, 2);
    end
end