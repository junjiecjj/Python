function BER = run_ISAC_FDM(alpha, SNR_dB)
% SNR_dB comes from compare script, can include negatives

rng(1);   % Fix random seed for reproducibility

% === Parameters (must match main_ISAC_FDM_waveform) ===
K = 64; cpLen = 16; M = 4;
Nt = 4; Nr = 4;
Nsym = 300;

Kc = round(alpha * K);
BER = zeros(1, length(SNR_dB));

for si = 1:length(SNR_dB)
    snr_db = SNR_dB(si);
    err = 0; total = 0;

    for n = 1:Nsym
        bits = randi([0 1], Kc*log2(M), 1);
        Xc = qammod(bits, M, ...
            'InputType','bit','UnitAveragePower',true);

        X = zeros(K,1);
        X(1:Kc) = Xc;   % FDM: lower band for comm

        s_t = ofdmMod(X, K, cpLen);

        H = mimoChannel(Nr, Nt);
        W = designPrecoder_gamma(H, 0);  % pure comm precoder

        x_tx = s_t * (W.');
        y_rx = awgn(x_tx * (H.'), snr_db, 'measured');

        Y = ofdmDemod(y_rx, K, cpLen);

        h_eff = H * W;
        denom = (h_eff' * h_eff);
        Y_eff = (Y * conj(h_eff)) / denom;

        Yc = Y_eff(1:Kc);
        bits_hat = qamdemod(Yc, M, ...
            'OutputType','bit','UnitAveragePower',true);

        [e, t] = berCount(bits, bits_hat);
        err = err + e;
        total = total + t;
    end

    ber = err / total;
    
    % -----------------------------------------------------------------
    % Minimum resolvable BER floor:
    % In finite-length Monte Carlo simulations, zero error events
    % may occur, leading to BER = 0. Such values cannot be displayed
    % on logarithmic-scale plots. Therefore, a minimum BER floor of
    % 1/(2*total) is applied for numerical stability and visualization
    % consistency. This does not affect the overall performance trend.
    % -----------------------------------------------------------------
    ber_floor = 1 / (2*total);
    ber = max(ber, ber_floor);
    
    BER(si) = ber;
end
end
