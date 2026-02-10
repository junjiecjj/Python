function BER = run_ISAC_OFDM(gamma, SNR_dB)
% SNR_dB comes from compare script, can include negatives

rng(1);   % Fix random seed for reproducibility

% === Parameters (must match main_ISAC_OFDM_waveform) ===
K = 64; cpLen = 16; M = 4;
Nt = 4; Nr = 4;
Nsym = 300;

BER = zeros(1, length(SNR_dB));

for si = 1:length(SNR_dB)
    snr_db = SNR_dB(si);
    err = 0; total = 0;

    for n = 1:Nsym
        bits = randi([0 1], K*log2(M), 1);
        X = qammod(bits, M, 'InputType','bit','UnitAveragePower',true);

        s_t = ofdmMod(X, K, cpLen);

        H = mimoChannel(Nr, Nt);
        W = designPrecoder_gamma(H, gamma);

        x_tx = s_t * (W.');
        y_rx = awgn(x_tx * (H.'), snr_db, 'measured');

        Y = ofdmDemod(y_rx, K, cpLen);

        h_eff = H * W;
        denom = (h_eff' * h_eff);
        Y_eff = (Y * conj(h_eff)) / denom;

        bits_hat = qamdemod(Y_eff, M, ...
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
