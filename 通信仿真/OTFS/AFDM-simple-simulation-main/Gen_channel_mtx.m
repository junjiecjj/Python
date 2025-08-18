% Copyright (c) 2025, Ruijie Zhang, University of Chinese Academy of Sciences
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
% 3. The reference listed below should be cited if the corresponding codes are used for
%   publication.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
%ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
%WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
%ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
%SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%    - Freely distributed for educational and research purposes

%  [R1]. Bemani, Ali, Nassar Ksairi, and Marios Kountouris. "Affine frequency division multiplexing for next generation wireless communications." IEEE Transactions on Wireless Communications 22.11 (2023): 8214-8229.


% Description: Generate delay-Doppler channel matrix, see equation (24) in [R1].
function [H] = Gen_channel_mtx(N, taps, chan_coef, delay_taps, Doppler_freq, c1)
Pi = [zeros(1,N-1) 1];
Pi = toeplitz([Pi(1) fliplr(Pi(2:end))], Pi);   % equation (25) in [R1]
H = zeros(N,N);
for i = 1:taps
    h_i = chan_coef(i);
    l_i = delay_taps(i);
    f_i = Doppler_freq(i);
    D_i = diag(exp(-1i*2*pi*f_i*(0:N-1)));
    for n = 0:N-1
        if n < l_i
            temp(n+1) = exp(-1i*2*pi*c1*(N^2 - 2*N*(l_i-n)));
        else
            temp(n+1) = 1;
        end
    end
    G_i = diag(temp);   % equation (26) in [R1]
    H = H + h_i * G_i * D_i * Pi^l_i;
end