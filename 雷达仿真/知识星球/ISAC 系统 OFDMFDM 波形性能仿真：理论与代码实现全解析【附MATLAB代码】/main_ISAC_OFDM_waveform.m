clc; clear; close all;

%% ========= System Parameters =========
K      = 64;      % number of subcarriers
cpLen  = 16;      % cyclic prefix length
M      = 4;       % QPSK
Nt     = 4;       % TX antennas
Nr     = 4;       % RX antennas
Nsym   = 800;     % number of OFDM symbols per SNR point

SNR_dB = [-10 -5 0 5 10 15 20 25 30];

gammaSet  = [0.5 0.7 0.8 0.9];

BER = zeros(length(gammaSet), length(SNR_dB));

%% ========= Main Loop =========
for gi = 1:length(gammaSet)
    gamma = gammaSet(gi);

    for si = 1:length(SNR_dB)
        snr_db = SNR_dB(si);

        err = 0; total = 0;

        for n = 1:Nsym
            % ----- Generate one OFDM symbol (single stream) -----
            bits = randi([0 1], K*log2(M), 1);
            X = qammod(bits, M, 'InputType','bit','UnitAveragePower',true); % Kx1 (freq)

            % ----- OFDM modulation: IFFT + CP -----
            s_t = ofdmMod(X, K, cpLen);   % (K+cpLen)x1 (time)

            % ----- MIMO channel + precoder (gamma in spatial domain) -----
            H = mimoChannel(Nr, Nt);                 % Nr x Nt
            W = designPrecoder_gamma(H, gamma);      % Nt x 1 (single-stream precoder)

            % Tx across antennas: (K+cp)xNt
            x_tx = s_t * (W.');                      % time waveform replicated & weighted

            % Rx: (K+cp)xNr
            y_rx = x_tx * (H.');                     % apply MIMO channel

            % ----- AWGN -----
            y_rx = awgn(y_rx, snr_db, 'measured');

            % ----- OFDM demod: remove CP + FFT -----
            Y = ofdmDemod(y_rx, K, cpLen);           % K x Nr (freq)

            % ----- Effective channel (single stream) -----
            h_eff = (H * W);                         % Nr x 1

            % ----- MRC combining across Rx antennas (single stream) -----
            % Y_eff[k] = (h_eff^H * y[k]) / (||h_eff||^2)
            denom = (h_eff' * h_eff);                % scalar
            Y_eff = (Y * conj(h_eff)) / denom;       % Kx1

            % ----- QPSK demod + BER -----
            bits_hat = qamdemod(Y_eff, M, 'OutputType','bit','UnitAveragePower',true);
            [e, t] = berCount(bits, bits_hat);

            err = err + e;
            total = total + t;
        end

        ber = err / total;
        
        % ===== 有限樣本下的 BER 下界（避免 semilogy 的 log(0)）=====
        if ber == 0
            ber = 0.5 / total;   % conservative finite-sample lower bound
        end
        
        BER(gi, si) = ber;
        
        fprintf('gamma=%.1f, SNR=%2d dB -> BER=%.4g\n', ...
                gamma, snr_db, BER(gi,si));
        
    end
end

%% ========= Plot =========
figure;
semilogy(SNR_dB, BER','-o','LineWidth',1.5);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
legend('\gamma=0.5','\gamma=0.7','\gamma=0.8','\gamma=0.9','Location','southwest');
title('ISAC-OFDM Waveform-level Simulation (Precoder-based \gamma)');
