function H_cold_new = draw_H_cold_realization(para, coldPara)
% Generate one random realization of the cold clutter channel tensor.
%
% Inputs:
%   coldPara   : Structure containing cold clutter parameters
%   para       : Structure containing system parameters
%
% Output:
%   H_cold_new : Cold clutter channel tensor of size Nr x Nt x L x N

N = para.N;              % Number of subcarriers
Nt = para.Nt;            % Number of transmit antennas
Nr = para.Nr;            % Number of receive antennas
L = para.L;              % Number of OFDM symbols / slow-time snapshots
Tsym = para.Tsym;        % OFDM symbol duration
deltaf = para.deltaf;    % Subcarrier spacing
fc = para.fc;            % Carrier frequency

% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);

% Draw frequency-correlated clutter coefficients
beta_cn_new = complex(zeros(coldPara.num, N));
for c = 1:coldPara.num
    beta_temp = coldPara.Rfreq_sqrt{c} * (randn(N, 1) + 1j * randn(N, 1)) / sqrt(2);
    beta_cn_new(c, :) = beta_temp.';
end

% Construct the cold clutter channel tensor
H_cold_new = complex(zeros(Nr, Nt, L, N));
parfor l = 1:L
    for n = 1:N
        temp_cold = 0;
        for c = 1:coldPara.num
            phase_cold = exp(1j * 2 * pi * (coldPara.dopplers(c) * (l-1) * Tsym - ...
                (n-1) * deltaf * coldPara.delays(c)));

            temp_cold = temp_cold + beta_cn_new(c, n) * phase_cold ...
                * b_n(coldPara.thetas(c), n-1) * a_n(coldPara.thetas(c), n-1)';
        end
        H_cold_new(:, :, l, n) = temp_cold;
    end
end
end