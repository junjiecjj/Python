function y_tar = echo_tar(para, tarPara, x_nl, a_n, b_n)
% Generate the target echo in the frequency-symbol domain.
%
% Inputs:
%   para    : Structure of basic system parameters
%   tarPara : Structure of target parameters
%   x_nl    : Transmit signal over antennas, subcarriers, and OFDM symbols
%   a_n     : Transmit steering-vector function handle
%   b_n     : Receive steering-vector function handle
%
% Output:
%   y_tar   : Target echo of size Nr x N x L

N = para.N;
L = para.L;
Nr = para.Nr;
Tsym = para.Tsym;
f_n = para.fc + (0:para.N-1) * para.deltaf;

y_tar = zeros(Nr, N, L);
for i = 1:Nr
    temp = 0;
    for m = 1:tarPara.M
        % Transmit-side projection along the target direction
        A_tx = a_n(tarPara.theta(m), 0:N-1);
        A_tx_expanded = repmat(A_tx, [1, 1, L]);
        s_nl = squeeze(sum(conj(A_tx_expanded) .* x_nl, 1));

        % Receive steering vector
        B_rx = b_n(tarPara.theta(m), 0:N-1);

        % Delay- and Doppler-dependent phase terms
        delay_vec = exp(-1j * 2 * pi * f_n' * tarPara.delays(m));
        dopp_vec = exp(-1j * 2 * pi * (tarPara.dopplers(m) * (0:L-1)' * Tsym));

        % Accumulate the echo contribution of the m-th target
        temp = temp + (tarPara.alpha_mn(m, 1:N).' .* B_rx(i, :).' .* delay_vec) ...
            * dopp_vec' .* s_nl;
    end
    y_tar(i, :, :) = temp;
end
end