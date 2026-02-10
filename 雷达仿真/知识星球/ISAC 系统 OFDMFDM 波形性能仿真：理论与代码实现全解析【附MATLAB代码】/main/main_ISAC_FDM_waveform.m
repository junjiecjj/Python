clc; clear; close all;

%% ========= System Parameters =========
K      = 64;      % total subcarriers
cpLen  = 16;
M      = 4;       % QPSK
Nt     = 4;
Nr     = 4;
Nsym   = 800;

SNR_dB = [-10 -5 0 5 10 15 20 25 30];

alphaSet = [0.5 0.7 0.8 0.9];   % bandwidth allocation (comm)

BER = zeros(length(alphaSet), length(SNR_dB));

%% ========= Main Loop =========
for ai = 1:length(alphaSet)
    alpha = alphaSet(ai);
    Kc = round(alpha * K);   % comm subcarriers

    for si = 1:length(SNR_dB)
        snr_db = SNR_dB(si);
        err = 0; total = 0;

        for n = 1:Nsym
            %% ----- Generate QPSK symbols for communication only -----
            bits = randi([0 1], Kc*log2(M), 1);
            Xc = qammod(bits, M, ...
                'InputType','bit','UnitAveragePower',true);

            %% ----- FDM subcarrier mapping -----
            X = zeros(K,1);       % total spectrum
            X(1:Kc) = Xc;         % lower band for communication
            % remaining subcarriers reserved for sensing

            %% ----- OFDM modulation -----
            s_t = ofdmMod(X, K, cpLen);

            %% ----- MIMO channel + precoder -----
            H = mimoChannel(Nr, Nt);
            W = designPrecoder_gamma(H, 0);  
            % gamma=0 : pure communication precoder

            x_tx = s_t * (W.');
            y_rx = x_tx * (H.');
            y_rx = awgn(y_rx, snr_db, 'measured');

            %% ----- OFDM demodulation -----
            Y = ofdmDemod(y_rx, K, cpLen);   % K x Nr

            %% ----- Effective channel -----
            h_eff = H * W;
            denom = (h_eff' * h_eff);
            Y_eff = (Y * conj(h_eff)) / denom;

            %% ----- Extract comm subcarriers only -----
            Yc = Y_eff(1:Kc);

            %% ----- Demodulation & BER -----
            bits_hat = qamdemod(Yc, M, ...
                'OutputType','bit','UnitAveragePower',true);

            [e, t] = berCount(bits, bits_hat);
            err = err + e;
            total = total + t;
        end

        ber = err / total;

        % ===== 避免 semilogy 無法顯示 BER = 0 =====
        if ber == 0
            ber = 0.5 / total;   % 保守下界（finite-sample lower bound）
        end
        
        BER(ai, si) = ber;
        
        fprintf('alpha=%.1f, SNR=%2d dB -> BER=%.4g\n', ...
                alpha, snr_db, BER(ai,si));

    end
end

%% ========= Plot =========
figure;
semilogy(SNR_dB, BER','-o','LineWidth',1.5);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
legend('\alpha=0.5','\alpha=0.7','\alpha=0.8','\alpha=0.9', ...
       'Location','southwest');
title('ISAC-FDM Waveform-level Simulation');
