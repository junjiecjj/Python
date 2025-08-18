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
%    - Freely distributed for educational and research purposes
%  [R1]. Bemani, Ali, Nassar Ksairi, and Marios Kountouris. "Affine frequency division multiplexing for next generation wireless communications." IEEE Transactions on Wireless Communications 22.11 (2023): 8214-8229.

clear; clc; close all;
rng(1)

%% System parameters %%
M_mod = 4;  % size of QAM constellation
N = 64;     % number of symbols(subcarriers)

car_fre = 4*10^9;   % carrier frequency
delta_f = 15*10^3;  % symbol spacing
T = 1/delta_f;      % symbol duration

eng_sqrt = (M_mod==2)+(M_mod~=2)*sqrt((M_mod-1)/6*(2^2));   % average power per symbol
SNR_dB = 0:2:16;    % set SNR here
SNR = 10.^(SNR_dB/10);
sigma_2 = (abs(eng_sqrt)^2)./SNR;   % noise power

N_frame = 10000;    % number of simulation frames

%% Generate synthetic delay-Doppler channel %%
taps = 9;       % number of paths
l_max = 3;      % maximum normalized delay index
k_max = 4;      % maximum normalized Doppler index
chan_coef = 1/sqrt(2).*(randn(1,taps)+1i.*randn(1,taps));   % follows Rayleigh distribution
delay_taps = randi(l_max, [1,taps]);
delay_taps = sort(delay_taps-min(delay_taps));      % integer delay shifts: random delays in range [0,l_max-1]
Doppler_taps = k_max-2*k_max*rand(1,taps);          % fractional Doppler shifts: uniformly distributed Doppler shifts in range [-k_max,k_max]
% Doppler_taps = round(Doppler_taps);     % cast to integer Doppler shifts
Doppler_freq = Doppler_taps/(N*T);      % f=k/(NT),f:Doppler shifts(Hz),k:normalized Doppler shifts

%% AFDM parameters %%
max_Doppler = max(Doppler_taps);
max_delay = max(delay_taps);

CPP_len = max_delay;    % CPP_len >= l_max-1
N_data = N-CPP_len;     % length of data symbols

k_v = 1;    % guard interval to combat fractional Doppler shifts, see equation (38) in [R1]
if (2*(max_Doppler+k_v)*(max_delay+1)+max_delay)>N_data
    error('subcarrier orthogonality is not satisfied');
end
c1 = (2*(max_Doppler+k_v)+1)/(2*N_data);    % equation (48) in [R1]
c2 = 1/(N_data^2);

%% Generate channel matrix %%
% discrete-time channel
L_set = unique(delay_taps);
gs=zeros(max_delay+1,N);      
for q=0:N-1
    for i=1:taps
        g_i=chan_coef(i);
        l_i=delay_taps(i);
        f_i=Doppler_freq(i);        
        gs(l_i+1,q+1)=gs(l_i+1,q+1)+g_i*exp(-1i*2*pi*f_i*q);  % equation (23) in [R1]
    end    
end

% channel matrix form
H = Gen_channel_mtx(N, taps, chan_coef, delay_taps, Doppler_freq, c1);  % equation (24) in [R1]
% Observe the structure of H
% imagesc(abs(H))

%% Start main simulation %%
for iesn0 = 1:length(SNR_dB)
    for iframe = 1:N_frame
        %% Tx data generation %% 
        x = randi([0, M_mod-1], N_data, 1);     % generate random bits
        x_qam = qammod(x, M_mod, 'gray');   % QAM modulation
        s = AFDM_mod(x_qam, c1, c2);    % AFDM modulation
        cpp = s(N_data-CPP_len:N_data-1).*exp(-1i*2*pi*c1*(N^2+2*N*(-CPP_len:-1).'));     % generate CPP
        s_cpp = [cpp; s];   % Insert CPP
        
        %% Through delay-Doppler channel %%
        r=zeros(N,1);
        for q=1:N
            for l=(L_set+1)
                if(q>=l)
                    r(q)=r(q)+gs(l,q)*s_cpp(q-l+1);  %equation (18) in [R1]
                end
            end
        end
        w = sqrt(sigma_2(iesn0)/2) * (randn(size(s_cpp)) + 1i*randn(size(s_cpp)));    % add Gaussian noise
        r=r+w;
        % r=H*s_cpp+w;  % or simply do this
        %% Rx detection %%
        x_est = H'/(H*H'+sigma_2(iesn0)*eye(N))*r;  % MMSE equalization, ideal channel estimation
        x_est_no_cpp = x_est(CPP_len+1:end);  % discard CPP
        y = AFDM_demod(x_est_no_cpp, c1, c2);  % AFDM demodulation
        x_est_bit = qamdemod(y, M_mod, 'gray');  % QAM demodulation
        %% Error count %%
        err_count(iframe) = sum(x_est_bit ~= x);    % calculate error bits
    end
    ber(iesn0) = sum(err_count)/length(x)/N_frame;  % calculate bit error rate
end
disp(ber)

%% Plot bit error rate %%
figure(1)
semilogy(SNR_dB, ber, 'b-o')
legend('MMSE equalization')
xlabel('SNR(dB)')
ylabel('BER')
title('BER of AFDM systems')
grid on