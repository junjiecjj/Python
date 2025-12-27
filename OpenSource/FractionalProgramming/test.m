
%.rtcContent { padding: 30px; } .lineNode {font-size: 16pt; font-family: Menlo, Monaco, Consolas, "Courier New", monospace; font-style: normal; font-weight: normal; }
% Comparing OFDM, GFDM, and FBMC under Multipath Fading with Radar Performance
clc;
clear;
close all;
% Parameters
N = 64;                 % Number of subcarriers
M = 4;                  % Modulation order (QPSK)
numSymbols = 1000;      % Number of symbols
SNR_dB = 0:2:20;       % SNR range in dB
numIter = 100;          % Number of iterations for averaging
% Multipath Channel
channel = [0.8; 0.4; 0.2]; % Simple multipath channel
% Radar Parameters
max_delay = 32;         % Maximum delay for ambiguity function
max_doppler = 0.1;      % Maximum Doppler frequency (normalized)
delay_bins = -max_delay:max_delay;
doppler_bins = linspace(-max_doppler, max_doppler, 64);
% Initialize BER arrays
BER_OFDM = zeros(length(SNR_dB), 1);
BER_GFDM = zeros(length(SNR_dB), 1);
BER_FBMC = zeros(length(SNR_dB), 1);
% Initialize Radar Performance arrays
range_resolution_OFDM = zeros(length(SNR_dB), 1);
range_resolution_GFDM = zeros(length(SNR_dB), 1);
range_resolution_FBMC = zeros(length(SNR_dB), 1);
doppler_resolution_OFDM = zeros(length(SNR_dB), 1);
doppler_resolution_GFDM = zeros(length(SNR_dB), 1);
doppler_resolution_FBMC = zeros(length(SNR_dB), 1);
PSL_OFDM = zeros(length(SNR_dB), 1); % Peak Sidelobe Level
PSL_GFDM = zeros(length(SNR_dB), 1);
PSL_FBMC = zeros(length(SNR_dB), 1);
% Main loop for SNR
for snrIdx = 1:length(SNR_dB)
    snr = SNR_dB(snrIdx);
    ber_ofdm = 0;
    ber_gfdm = 0;
    ber_fbmc = 0;

    % Initialize radar metrics for current SNR
    range_res_ofdm = 0;
    range_res_gfdm = 0;
    range_res_fbmc = 0;

    doppler_res_ofdm = 0;
    doppler_res_gfdm = 0;
    doppler_res_fbmc = 0;

    psl_ofdm = 0;
    psl_gfdm = 0;
    psl_fbmc = 0;

    for iter = 1:numIter
        % Generate random data
        data = randi([0 M-1], N, numSymbols);

        % QPSK Modulation
        modData = pskmod(data(:), M, pi/4);
        modData = reshape(modData, N, numSymbols);

        % OFDM
        ofdmSymbols = ifft(modData, N);
        cp_len = N/4;
        ofdmSymbols_with_cp = [ofdmSymbols(end-cp_len+1:end, :); ofdmSymbols]; % Add CP
        ofdmSignal = ofdmSymbols_with_cp(:);

        % GFDM
        gfdmSymbols = gfdm_modulate(modData, N, M);
        gfdmSignal = gfdmSymbols(:);

        % FBMC
        fbmcSymbols = fbmc_modulate(modData, N, M);
        fbmcSignal = fbmcSymbols(:);

        % Pass through multipath channel
        ofdmRx = filter(channel, 1, ofdmSignal);
        gfdmRx = filter(channel, 1, gfdmSignal);
        fbmcRx = filter(channel, 1, fbmcSignal);

        % Add AWGN noise
        ofdmRx = awgn(ofdmRx, snr, 'measured');
        gfdmRx = awgn(gfdmRx, snr, 'measured');
        fbmcRx = awgn(fbmcRx, snr, 'measured');

        % OFDM Receiver
        ofdmRx_reshaped = reshape(ofdmRx, N + cp_len, numSymbols);
        ofdmRx_no_cp = ofdmRx_reshaped(cp_len+1:end, :); % Remove CP
        ofdmRx_demod = fft(ofdmRx_no_cp, N);

        % GFDM Receiver
        gfdmRx_reshaped = reshape(gfdmRx, N, numSymbols);
        gfdmRx_demod = gfdm_demodulate(gfdmRx_reshaped, N, M);

        % FBMC Receiver
        fbmcRx_reshaped = reshape(fbmcRx, N, numSymbols);
        fbmcRx_demod = fbmc_demodulate(fbmcRx_reshaped, N, M);

        % QPSK Demodulation
        ofdmData = pskdemod(ofdmRx_demod(:), M, pi/4);
        gfdmData = pskdemod(gfdmRx_demod(:), M, pi/4);
        fbmcData = pskdemod(fbmcRx_demod(:), M, pi/4);

        % Calculate BER
        [~, ber_ofdm_iter] = biterr(data(:), ofdmData);
        [~, ber_gfdm_iter] = biterr(data(:), gfdmData);
        [~, ber_fbmc_iter] = biterr(data(:), fbmcData);

        ber_ofdm = ber_ofdm + ber_ofdm_iter;
        ber_gfdm = ber_gfdm + ber_gfdm_iter;
        ber_fbmc = ber_fbmc + ber_fbmc_iter;

        % Calculate Radar Ambiguity Functions
        amb_ofdm = calculate_ambiguity_function(ofdmSignal(1:min(512,length(ofdmSignal))), delay_bins, doppler_bins);
        amb_gfdm = calculate_ambiguity_function(gfdmSignal(1:min(512,length(gfdmSignal))), delay_bins, doppler_bins);
        amb_fbmc = calculate_ambiguity_function(fbmcSignal(1:min(512,length(fbmcSignal))), delay_bins, doppler_bins);

        % Extract Radar Performance Metrics
        [range_res_iter_ofdm, doppler_res_iter_ofdm, psl_iter_ofdm] = extract_radar_metrics(amb_ofdm, delay_bins, doppler_bins);
        [range_res_iter_gfdm, doppler_res_iter_gfdm, psl_iter_gfdm] = extract_radar_metrics(amb_gfdm, delay_bins, doppler_bins);
        [range_res_iter_fbmc, doppler_res_iter_fbmc, psl_iter_fbmc] = extract_radar_metrics(amb_fbmc, delay_bins, doppler_bins);

        range_res_ofdm = range_res_ofdm + range_res_iter_ofdm;
        range_res_gfdm = range_res_gfdm + range_res_iter_gfdm;
        range_res_fbmc = range_res_fbmc + range_res_iter_fbmc;

        doppler_res_ofdm = doppler_res_ofdm + doppler_res_iter_ofdm;
        doppler_res_gfdm = doppler_res_gfdm + doppler_res_iter_gfdm;
        doppler_res_fbmc = doppler_res_fbmc + doppler_res_iter_fbmc;

        psl_ofdm = psl_ofdm + psl_iter_ofdm;
        psl_gfdm = psl_gfdm + psl_iter_gfdm;
        psl_fbmc = psl_fbmc + psl_iter_fbmc;
    end

    % Average BER
    BER_OFDM(snrIdx) = ber_ofdm / numIter;
    BER_GFDM(snrIdx) = ber_gfdm / numIter;
    BER_FBMC(snrIdx) = ber_fbmc / numIter;

    % Average Radar Metrics
    range_resolution_OFDM(snrIdx) = range_res_ofdm / numIter;
    range_resolution_GFDM(snrIdx) = range_res_gfdm / numIter;
    range_resolution_FBMC(snrIdx) = range_res_fbmc / numIter;

    doppler_resolution_OFDM(snrIdx) = doppler_res_ofdm / numIter;
    doppler_resolution_GFDM(snrIdx) = doppler_res_gfdm / numIter;
    doppler_resolution_FBMC(snrIdx) = doppler_res_fbmc / numIter;

    PSL_OFDM(snrIdx) = psl_ofdm / numIter;
    PSL_GFDM(snrIdx) = psl_gfdm / numIter;
    PSL_FBMC(snrIdx) = psl_fbmc / numIter;
end
% Plot BER vs SNR
figure;
semilogy(SNR_dB, BER_OFDM, 'bo-', 'LineWidth', 2);
hold on;
semilogy(SNR_dB, BER_GFDM, 'rs-', 'LineWidth', 2);
semilogy(SNR_dB, BER_FBMC, 'gd-', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
legend('OFDM', 'GFDM', 'FBMC');
title('BER Comparison of OFDM, GFDM, and FBMC under Multipath Fading');
% Plot Radar Performance Metrics
figure;
% Range Resolution
subplot(2,2,1);
plot(SNR_dB, range_resolution_OFDM, 'bo-', 'LineWidth', 2);
hold on;
plot(SNR_dB, range_resolution_GFDM, 'rs-', 'LineWidth', 2);
plot(SNR_dB, range_resolution_FBMC, 'gd-', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Range Resolution (samples)');
legend('OFDM', 'GFDM', 'FBMC');
title('Range Resolution vs SNR');
% Doppler Resolution
subplot(2,2,2);
plot(SNR_dB, doppler_resolution_OFDM, 'bo-', 'LineWidth', 2);
hold on;
plot(SNR_dB, doppler_resolution_GFDM, 'rs-', 'LineWidth', 2);
plot(SNR_dB, doppler_resolution_FBMC, 'gd-', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Doppler Resolution (normalized)');
legend('OFDM', 'GFDM', 'FBMC');
title('Doppler Resolution vs SNR');
% Peak Sidelobe Level
subplot(2,2,3);
plot(SNR_dB, PSL_OFDM, 'bo-', 'LineWidth', 2);
hold on;
plot(SNR_dB, PSL_GFDM, 'rs-', 'LineWidth', 2);
plot(SNR_dB, PSL_FBMC, 'gd-', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('Peak Sidelobe Level (dB)');
legend('OFDM', 'GFDM', 'FBMC');
title('Peak Sidelobe Level vs SNR');
% Plot Ambiguity Functions at high SNR
high_snr_idx = length(SNR_dB);
plot_ambiguity_functions(ofdmSignal, gfdmSignal, fbmcSignal, delay_bins, doppler_bins);
% GFDM Modulation Function
function gfdmSymbols = gfdm_modulate(data, N, M)
    % Improved GFDM modulation with circular filtering
    K = 7; % Number of subsymbols
    gfdmSymbols = zeros(N, size(data,2));
    for k = 1:K
        % Apply circular shift and filter
        shift_data = circshift(data, k-1, 1);
        filtered = ifft(shift_data, N) .* exp(1j*2*pi*(k-1)*(0:N-1)'/N);
        gfdmSymbols = gfdmSymbols + filtered;
    end
    gfdmSymbols = gfdmSymbols / sqrt(K);
end
% GFDM Demodulation Function
function gfdmData = gfdm_demodulate(gfdmRx, N, M)
    % Improved GFDM demodulation
    K = 7; % Number of subsymbols
    gfdmData = zeros(size(gfdmRx));
    for k = 1:K
        filtered = fft(gfdmRx .* exp(-1j*2*pi*(k-1)*(0:N-1)'/N), N);
        gfdmData = gfdmData + circshift(filtered, -(k-1), 1);
    end
    gfdmData = gfdmData / sqrt(K);
end
% FBMC Modulation Function
function fbmcSymbols = fbmc_modulate(data, N, M)
    % Improved FBMC modulation with prototype filter
    fbmcSymbols = ifft(data, N);
    % Apply prototype filter (simplified)
    prototype_filter = sqrt(2)*sin(pi*(0:N-1)'/(N-1));
    fbmcSymbols = fbmcSymbols .* prototype_filter;
end
% FBMC Demodulation Function
function fbmcData = fbmc_demodulate(fbmcRx, N, M)
    % Improved FBMC demodulation
    prototype_filter = sqrt(2)*sin(pi*(0:N-1)'/(N-1));
    fbmcData = fft(fbmcRx .* prototype_filter, N);
end
% Ambiguity Function Calculation
function amb = calculate_ambiguity_function(signal, delay_bins, doppler_bins)
    L = length(signal);
    amb = zeros(length(doppler_bins), length(delay_bins));

    % 确保信号是列向量
    signal = signal(:);

    for i = 1:length(delay_bins)
        delay = delay_bins(i);

        for j = 1:length(doppler_bins)
            doppler = doppler_bins(j);

            % 处理延迟
            if delay >= 0
                % 正延迟：信号1从开始到L-delay，信号2从delay+1到结束
                if delay < L
                    signal1 = signal(1:end-delay);
                    signal2 = signal(delay+1:end);
                else
                    signal1 = [];
                    signal2 = [];
                end
            else
                % 负延迟：交换信号
                delay = abs(delay);
                if delay < L
                    signal1 = signal(delay+1:end);
                    signal2 = signal(1:end-delay);
                else
                    signal1 = [];
                    signal2 = [];
                end
            end

            if ~isempty(signal1) && ~isempty(signal2) && length(signal1) == length(signal2)
                % 应用多普勒频移
                t = (0:length(signal1)-1)';
                doppler_phase = exp(1j*2*pi*doppler*t);

                % 计算互相关
                corr_val = sum(signal1 .* conj(signal2) .* doppler_phase);
                amb(j,i) = abs(corr_val);
            else
                amb(j,i) = 0;
            end
        end
    end

    % 归一化
    if max(amb(:)) > 0
        amb = amb / max(amb(:));
    end
end
% Extract Radar Performance Metrics
function [range_res, doppler_res, psl] = extract_radar_metrics(amb, delay_bins, doppler_bins)
    % 获取ambiguity function的尺寸
    [num_doppler, num_delay] = size(amb);

    % Range resolution (使用零多普勒剖面)
    zero_doppler_profile = amb(doppler_bins == 0, :);
    if isempty(zero_doppler_profile)
        % 如果没有精确匹配，使用最接近零多普勒的剖面
        [~, zero_idx] = min(abs(doppler_bins));
        zero_doppler_profile = amb(zero_idx, :);
    end

    % 找到主瓣峰值
    [max_val, max_idx] = max(zero_doppler_profile);

    % 寻找-3dB点
    threshold = max_val * 10^(-3/20); % -3dB阈值

    % 向左寻找-3dB点
    left_idx = max_idx;
    while left_idx > 1 && zero_doppler_profile(left_idx) > threshold
        left_idx = left_idx - 1;
    end

    % 向右寻找-3dB点
    right_idx = max_idx;
    while right_idx < num_delay && zero_doppler_profile(right_idx) > threshold
        right_idx = right_idx + 1;
    end

    % 计算范围分辨率（延迟方向的-3dB宽度）
    if left_idx >= 1 && right_idx <= num_delay
        range_res = abs(delay_bins(right_idx) - delay_bins(left_idx));
    else
        range_res = abs(delay_bins(2) - delay_bins(1)) * 4; % 默认值
    end

    % Doppler resolution (使用零延迟剖面)
    zero_delay_profile = amb(:, delay_bins == 0);
    if isempty(zero_delay_profile)
        % 如果没有精确匹配，使用最接近零延迟的剖面
        [~, zero_idx] = min(abs(delay_bins));
        zero_delay_profile = amb(:, zero_idx);
    end

    % 找到主瓣峰值
    [max_val, max_idx] = max(zero_delay_profile);

    % 寻找-3dB点
    threshold = max_val * 10^(-3/20); % -3dB阈值

    % 向下寻找-3dB点
    down_idx = max_idx;
    while down_idx > 1 && zero_delay_profile(down_idx) > threshold
        down_idx = down_idx - 1;
    end

    % 向上寻找-3dB点
    up_idx = max_idx;
    while up_idx < num_doppler && zero_delay_profile(up_idx) > threshold
        up_idx = up_idx + 1;
    end

    % 计算多普勒分辨率（多普勒方向的-3dB宽度）
    if down_idx >= 1 && up_idx <= num_doppler
        doppler_res = abs(doppler_bins(up_idx) - doppler_bins(down_idx));
    else
        doppler_res = abs(doppler_bins(2) - doppler_bins(1)) * 4; % 默认值
    end

    % Peak Sidelobe Level
    [max_val, max_idx] = max(amb(:));
    [max_row, max_col] = ind2sub(size(amb), max_idx);

    % 创建主瓣掩码（排除主瓣区域）
    main_lobe_region = 3; % 主瓣区域大小
    row_range = max(1, max_row-main_lobe_region):min(num_doppler, max_row+main_lobe_region);
    col_range = max(1, max_col-main_lobe_region):min(num_delay, max_col+main_lobe_region);

    % 将主瓣区域设置为最小值
    sidelobes = amb;
    sidelobes(row_range, col_range) = 0;

    % 找到最大旁瓣
    max_sidelobe = max(sidelobes(:));

    if max_sidelobe > 0
        psl = 20*log10(max_sidelobe / max_val);
    else
        psl = -100; % 如果没有旁瓣，设置为很低的值
    end
end
% Plot Ambiguity Functions
function plot_ambiguity_functions(ofdm_signal, gfdm_signal, fbmc_signal, delay_bins, doppler_bins)
    % 使用固定长度截取信号
    sig_len = 256; % 固定长度

    % 确保信号足够长
    ofdm_segment = ofdm_signal(1:min(sig_len, length(ofdm_signal)));
    gfdm_segment = gfdm_signal(1:min(sig_len, length(gfdm_signal)));
    fbmc_segment = fbmc_signal(1:min(sig_len, length(fbmc_signal)));

    % 如果信号太短，进行填充
    if length(ofdm_segment) < sig_len
        ofdm_segment = [ofdm_segment; zeros(sig_len - length(ofdm_segment), 1)];
    end
    if length(gfdm_segment) < sig_len
        gfdm_segment = [gfdm_segment; zeros(sig_len - length(gfdm_segment), 1)];
    end
    if length(fbmc_segment) < sig_len
        fbmc_segment = [fbmc_segment; zeros(sig_len - length(fbmc_segment), 1)];
    end

    % 计算ambiguity functions
    amb_ofdm = calculate_ambiguity_function(ofdm_segment, delay_bins, doppler_bins);
    amb_gfdm = calculate_ambiguity_function(gfdm_segment, delay_bins, doppler_bins);
    amb_fbmc = calculate_ambiguity_function(fbmc_segment, delay_bins, doppler_bins);

    figure;

    % OFDM Ambiguity Function
    subplot(1,3,1);
    imagesc(delay_bins, doppler_bins, 20*log10(max(amb_ofdm, 1e-6))); % 避免log(0)
    axis xy; colorbar;
    xlabel('Delay (samples)');
    ylabel('Doppler (normalized)');
    title('OFDM Ambiguity Function');
    clim([-40 0]);

    % GFDM Ambiguity Function
    subplot(1,3,2);
    imagesc(delay_bins, doppler_bins, 20*log10(max(amb_gfdm, 1e-6)));
    axis xy; colorbar;
    xlabel('Delay (samples)');
    ylabel('Doppler (normalized)');
    title('GFDM Ambiguity Function');
    clim([-40 0]);

    % FBMC Ambiguity Function
    subplot(1,3,3);
    imagesc(delay_bins, doppler_bins, 20*log10(max(amb_fbmc, 1e-6)));
    axis xy; colorbar;
    xlabel('Delay (samples)');
    ylabel('Doppler (normalized)');
    title('FBMC Ambiguity Function');
    clim([-40 0]);

    sgtitle('Radar Ambiguity Functions Comparison');
end