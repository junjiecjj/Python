%% ========================================================================
% main_isac_compare_BER_NMSE_SNR.m
% ------------------------------------------------------------------------
% 基于已有函数的单脚本 Monte Carlo 仿真：
% 1) 通信 BER 随 SNR 变化
% 2) 目标距离估计 NMSE 随 SNR 变化
% 3) 目标速度估计 NMSE 随 SNR 变化
%
% 依赖当前文件夹中的已有函数：
%   get_waveform_operator.m
%   build_delay_doppler_channel.m
%   simulate_comm_link.m
%   simulate_radar_link.m
%   qpsk_mod.m
%   qpsk_hard_demod.m
%
% 说明：
% - 四种信号：RM / OFDM / OTFS / AFDM
% - 感知端：基站缓存已知发射样本，利用 CAF 取双目标峰值
% - NMSE 采用物理距离 / 物理速度定义
% ========================================================================

clc; clear; close all;

%默认参数

params = struct();


% OTFS / RM / AFDM / OFDM 统一块长
params.M = 16;
params.N = 16;
params.L = params.M * params.N;

% ------------------------------
% 感知双目标离散时延-多普勒参数
% ------------------------------
params.target.delays   = [5, 18];
params.target.dopplers = [3, -4];
params.target.gains    = [0.68 * exp(1j*0.2),0.33 * exp(1j*0.8)];

% ------------------------------
% 通信用户离散时延-多普勒信道
% delays/dopplers 均按“离散bin”设置
% ------------------------------
params.user.delays   = [0, 3, 11, 17];
params.user.dopplers = [0, 1, -2, 3];
params.user.gains    = [1.00 * exp(1j*0.0), 0.75 * exp(1j*0.4),0.55 * exp(-1j*0.9),0.38 * exp(1j*1.2)];

% params.user.delays   = params.target.delays;
% params.user.dopplers = params.target.dopplers;
% params.user.gains    = params.target.gains;

% CAF 搜索网格
params.delayGrid   = 0:24;
params.dopplerGrid = -8:8;

% AFDM 参数
params.afdm.c1 = 5 / params.L;
params.afdm.c2 = 1 / (params.L^2);

% RM 随机置换种子（单独固定，便于复现）
params.rmPermutationSeed = 314159;

L = params.M * params.N;

M = params.M;
N = params.N;

waveformList = {'RM','OFDM','OTFS','AFDM'};
numWave = numel(waveformList);

snrVec = -10:5:20;
numSNR = numel(snrVec);

numMonte = 1000;              % Monte Carlo 次数，可改大
guardDelay = 1;              % CAF 峰值保护区：时延方向
guardDoppler = 1;            % CAF 峰值保护区：多普勒方向

% 物理参数：用于 delay / Doppler bin 映射成 距离 / 速度
c  = 3e8;                    % 光速
fs = 20e6;                   % 采样率
fc = 24e9;                   % 载频
lambda = c / fc;
Ts = 1 / fs;

rangeRes = c * Ts / 2;               % 每个时延 bin 对应的距离分辨率
velRes   = lambda * fs / (2 * L);    % 每个多普勒 bin 对应的速度分辨率

trueDelays   = params.target.delays(:).';
trueDopplers = params.target.dopplers(:).';
numTargets   = numel(trueDelays);

trueRanges = trueDelays * rangeRes;
trueVels   = trueDopplers * velRes;

% 为了消除目标排列不确定性，按真实距离升序固定参考顺序
[trueRangesSorted, idxTrueSort] = sort(trueRanges, 'ascend');
trueVelsSorted = trueVels(idxTrueSort);

%% -------------------- 构造通信/感知信道 --------------------
H_comm = build_delay_doppler_channel( ...
    L, params.user.delays, params.user.dopplers, params.user.gains);

H_radar = build_delay_doppler_channel( ...
    L, params.target.delays, params.target.dopplers, params.target.gains);

%% -------------------- 预生成各波形算子 --------------------
ops = cell(numWave,1);
for k = 1:numWave
    ops{k} = get_waveform_operator(waveformList{k}, params);
end

%% -------------------- 结果变量 --------------------
BER_mat        = zeros(numWave, numSNR);
RangeNMSE_mat  = zeros(numWave, numSNR);
VelNMSE_mat    = zeros(numWave, numSNR);

%% -------------------- Monte Carlo 主循环 --------------------
fprintf('================ Monte Carlo 仿真开始 ================\n');
fprintf('L = %d, M = %d, N = %d\n', L, M, N);
fprintf('SNR 扫描 = [%s] dB\n', num2str(snrVec));
fprintf('Monte Carlo 次数 = %d\n', numMonte);
fprintf('=====================================================\n\n');

for iSNR = 1:numSNR
    snrDb = snrVec(iSNR);
    
    fprintf('---- SNR = %2d dB ----\n', snrDb);

    bitErrCnt      = zeros(numWave,1);
    bitTotalCnt    = zeros(numWave,1);
    rangeNMSEAcc   = zeros(numWave,1);
    velNMSEAcc     = zeros(numWave,1);

    rng(1,"twister");

    for imc = 1:numMonte
        % 同一次 Monte Carlo 中，四种波形使用同一组信息比特
        bitTx = randi([0,1], 2*L, 1);
        symTx = qpsk_mod(bitTx);

        for k = 1:numWave
            op = ops{k};

            % ---------------- 通信链路 ----------------
            commRes = simulate_comm_link(symTx, bitTx, op, H_comm, snrDb);

            bitErrCnt(k)   = bitErrCnt(k) + sum(commRes.bitHat ~= bitTx);
            bitTotalCnt(k) = bitTotalCnt(k) + numel(bitTx);

            % ---------------- 感知链路 ----------------
            radarRes = simulate_radar_link( ...
                commRes.txSamples, H_radar, snrDb, ...
                params.delayGrid, params.dopplerGrid);

            CAFabs = abs(radarRes.CAF);
            CAFwork = CAFabs;

            estDelays   = zeros(1, numTargets);
            estDopplers = zeros(1, numTargets);

            % 从 CAF 中提取前 numTargets 个峰值，并做简单保护区抑制
            for it = 1:numTargets
                [~, idxMax] = max(CAFwork(:));
                [rowMax, colMax] = ind2sub(size(CAFwork), idxMax);

                estDopplers(it) = params.dopplerGrid(rowMax);
                estDelays(it)   = params.delayGrid(colMax);

                r1 = max(1, rowMax - guardDoppler);
                r2 = min(size(CAFwork,1), rowMax + guardDoppler);
                c1 = max(1, colMax - guardDelay);
                c2 = min(size(CAFwork,2), colMax + guardDelay);

                CAFwork(r1:r2, c1:c2) = -inf;
            end

            % 转换为物理距离 / 速度
            estRanges = estDelays * rangeRes;
            estVels   = estDopplers * velRes;

            % 与真实目标进行最优匹配（这里只考虑双目标；若目标数更多可扩展）
            if numTargets == 2
                estRanges_case1 = estRanges([1 2]);
                estVels_case1   = estVels([1 2]);

                estRanges_case2 = estRanges([2 1]);
                estVels_case2   = estVels([2 1]);

                cost1 = norm(estRanges_case1 - trueRangesSorted)^2 / (norm(trueRangesSorted)^2 + eps) + ...
                        norm(estVels_case1   - trueVelsSorted)^2   / (norm(trueVelsSorted)^2 + eps);

                cost2 = norm(estRanges_case2 - trueRangesSorted)^2 / (norm(trueRangesSorted)^2 + eps) + ...
                        norm(estVels_case2   - trueVelsSorted)^2   / (norm(trueVelsSorted)^2 + eps);

                if cost1 <= cost2
                    estRangesMatched = estRanges_case1;
                    estVelsMatched   = estVels_case1;
                else
                    estRangesMatched = estRanges_case2;
                    estVelsMatched   = estVels_case2;
                end
            else
                % 若目标数不是2，则简单按估计距离排序
                [estRangesMatched, idxEstSort] = sort(estRanges, 'ascend');
                estVelsMatched = estVels(idxEstSort);
            end

            % 计算距离 NMSE / 速度 NMSE
            rangeNMSE_this = norm(estRangesMatched - trueRangesSorted)^2 / (norm(trueRangesSorted)^2 + eps);
            velNMSE_this   = norm(estVelsMatched   - trueVelsSorted)^2   / (norm(trueVelsSorted)^2 + eps);

            rangeNMSEAcc(k) = rangeNMSEAcc(k) + rangeNMSE_this;
            velNMSEAcc(k)   = velNMSEAcc(k)   + velNMSE_this;
        end
    end

    % 当前 SNR 下的平均结果
    BER_mat(:, iSNR)       = bitErrCnt ./ bitTotalCnt;
    RangeNMSE_mat(:, iSNR) = rangeNMSEAcc / numMonte;
    VelNMSE_mat(:, iSNR)   = velNMSEAcc   / numMonte;

    for k = 1:numWave
        fprintf('%-5s : BER = %.4e, Range-NMSE = %.4e, Vel-NMSE = %.4e\n', ...
            waveformList{k}, BER_mat(k,iSNR), RangeNMSE_mat(k,iSNR), VelNMSE_mat(k,iSNR));
    end
    fprintf('\n');
end

%% -------------------- 结果作图：图1 BER-SNR --------------------
clr = lines(numWave);
mk  = {'o-','s-','d-','^-'};
lw  = 1.8;
ms  = 7;

figure('Color','w');
hold on; grid on; box on;
for k = 1:numWave
    semilogy(snrVec, BER_mat(k,:), mk{k}, ...
        'Color', clr(k,:), 'LineWidth', lw, 'MarkerSize', ms, ...
        'DisplayName', waveformList{k});
end
xlabel('SNR (dB)');
ylabel('BER');
title('通信 BER 随 SNR 变化');
legend('Location','southwest');
% ylim([1e-5 1]);

%% -------------------- 结果作图：图2 距离NMSE-SNR --------------------
figure('Color','w');
hold on; grid on; box on;
for k = 1:numWave
    semilogy(snrVec, RangeNMSE_mat(k,:), mk{k}, ...
        'Color', clr(k,:), 'LineWidth', lw, 'MarkerSize', ms, ...
        'DisplayName', waveformList{k});
end
xlabel('SNR (dB)');
ylabel('距离估计 NMSE');
title('估计的目标距离 NMSE 随 SNR 变化');
legend('Location','southwest');

%% -------------------- 结果作图：图3 速度NMSE-SNR --------------------
figure('Color','w');
hold on; grid on; box on;
for k = 1:numWave
    semilogy(snrVec, VelNMSE_mat(k,:), mk{k}, ...
        'Color', clr(k,:), 'LineWidth', lw, 'MarkerSize', ms, ...
        'DisplayName', waveformList{k});
end
xlabel('SNR (dB)');
ylabel('速度估计 NMSE');
title('估计的目标速度 NMSE 随 SNR 变化');
legend('Location','southwest');

%% -------------------- 保存结果到工作区 --------------------
results_snr = struct();
results_snr.waveformList   = waveformList;
results_snr.snrVec         = snrVec;
results_snr.numMonte       = numMonte;
results_snr.BER_mat        = BER_mat;
results_snr.RangeNMSE_mat  = RangeNMSE_mat;
results_snr.VelNMSE_mat    = VelNMSE_mat;
results_snr.trueRanges     = trueRangesSorted;
results_snr.trueVels       = trueVelsSorted;
results_snr.rangeRes       = rangeRes;
results_snr.velRes         = velRes;
results_snr.H_comm         = H_comm;
results_snr.H_radar        = H_radar;

assignin('base', 'params_isac_demo_snr', params);
assignin('base', 'results_isac_demo_snr', results_snr);

fprintf('================ 仿真结束 ================\n');
fprintf('结果已写入工作区：results_isac_demo_snr\n');
fprintf('参数已写入工作区：params_isac_demo_snr\n');
fprintf('==========================================\n');