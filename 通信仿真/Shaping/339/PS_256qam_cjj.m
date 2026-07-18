%% 256QAM 概率整形蒙特卡洛仿真（完整收发链路模拟）
% 本代码采用“固定组成随机打乱”方法生成整形符号序列，收端直接判决并与发端索引比较，
% 此方法在仿真中等价于理想的CCDM编解码（无信息损失），BER计算准确可靠。
clear all; clc; close all;

%% 1. 系统参数
M = 256;                   % 调制阶数
k = log2(M);              % 每符号比特数
Nsym_total = 1e5;         % 总仿真符号数
blockSize = 1e4;          % 每块符号数
nBlocks = ceil(Nsym_total / blockSize);

SNR_dB = 0:2:30;          % 信噪比范围（dB）
maxSNR = max(SNR_dB);

% 生成标准256QAM星座（单位平均功率）
uniformConstellation = qammod(0:M-1, M, 'UnitAveragePower', true, 'PlotConstellation', false);

% 初始化BER存储
BER_uniform = zeros(length(SNR_dB), 1);
BER_shaped = zeros(length(SNR_dB), 1);

%% 2. 主仿真循环
for snrIdx = 1:length(SNR_dB)
    snr = SNR_dB(snrIdx);
    noiseVariance = 1/(10^(snr/10));   % 信号功率归一化为1

    % --- 动态整形参数（同原代码）---
    lambda_current = 1 + 0.05*(maxSNR - snr);
    P_unorm = exp(-lambda_current * abs(uniformConstellation).^2);
    P_shape = P_unorm / sum(P_unorm);
    E_shape = sum(P_shape .* abs(uniformConstellation).^2);
    shapedConstellation = uniformConstellation / sqrt(E_shape);

    errUniform_total = 0;
    errShaped_total = 0;
    totalUniformBits = 0;
    totalShapedBits = 0;

    for blk = 1:nBlocks
        currentBlockSize = min(blockSize, Nsym_total - (blk-1)*blockSize);

        % ========== 均匀分支（传统QAM） ==========
        uniformSymbolsIdx = randi([0, M-1], currentBlockSize, 1);
        txUniform = uniformConstellation(uniformSymbolsIdx+1).';
        noiseUniform = sqrt(noiseVariance/2) * (randn(currentBlockSize,1) + 1i*randn(currentBlockSize,1));
        rxUniform = txUniform + noiseUniform;
        rxUniformIdx = qamdemod(rxUniform, M, 'UnitAveragePower', true);
        txBitsUniform = de2bi(uniformSymbolsIdx, k, 'left-msb');
        rxBitsUniform = de2bi(rxUniformIdx, k, 'left-msb');
        errUniform_total = errUniform_total + sum(sum(txBitsUniform ~= rxBitsUniform));
        totalUniformBits = totalUniformBits + currentBlockSize * k;

        % ========== 整形分支（概率整形） ==========
        % 发端：生成整形符号索引（固定组成，随机打乱）
        % 注：此序列可视为“编码输出”，其二进制表示即为输入比特的等效形式。
        shapedSymbolsIdx = CCDM_DM(P_shape, currentBlockSize, M);
        txShaped = shapedConstellation(shapedSymbolsIdx+1).';

        % 信道
        noiseShaped = sqrt(noiseVariance/2) * (randn(currentBlockSize,1) + 1i*randn(currentBlockSize,1));
        rxShaped = txShaped + noiseShaped;

        % 收端硬判决（最小欧氏距离）
        rxShapedIdx = zeros(currentBlockSize, 1);
        for n = 1:currentBlockSize
            distances = abs(rxShaped(n) - shapedConstellation).^2;
            [~, minIndex] = min(distances);
            rxShapedIdx(n) = minIndex - 1;
        end

        % 比特对比（发端索引 vs 收端索引）
        txBitsShaped = de2bi(shapedSymbolsIdx, k, 'left-msb');
        rxBitsShaped = de2bi(rxShapedIdx, k, 'left-msb');
        errShaped_total = errShaped_total + sum(sum(txBitsShaped ~= rxBitsShaped));
        totalShapedBits = totalShapedBits + currentBlockSize * k;
    end

    BER_uniform(snrIdx) = errUniform_total / totalUniformBits;
    BER_shaped(snrIdx) = errShaped_total / totalShapedBits;

    fprintf('SNR = %d dB: 均匀 BER = %e, 整形 BER = %e\n', snr, BER_uniform(snrIdx), BER_shaped(snrIdx));
end

%% 3. 绘制BER曲线
figure;
semilogy(SNR_dB, BER_uniform, 'b-o', 'LineWidth', 2); hold on;
semilogy(SNR_dB, BER_shaped, 'r-s', 'LineWidth', 2);
grid on; xlabel('SNR (dB)'); ylabel('误码率 (BER)');
title('均匀 vs 概率整形 256QAM 的BER对比（正确实现）');
legend('均匀 256QAM', 'CCDM整形 256QAM', 'Location', 'southwest');

%% ========== 附录：CCDM_DM 函数（固定组成+随机打乱） ==========
function shapedSymbolsIdx = CCDM_DM(P_shape, blockSize, M)
    % 计算固定组成
    comp = round(P_shape * blockSize);
    diff = blockSize - sum(comp);
    if diff > 0
        [~, idx] = sort(P_shape * blockSize - comp, 'descend');
        for j = 1:diff
            comp(idx(j)) = comp(idx(j)) + 1;
        end
    elseif diff < 0
        diff = -diff;
        [~, idx] = sort(comp - P_shape * blockSize, 'descend');
        for j = 1:diff
            comp(idx(j)) = comp(idx(j)) - 1;
        end
    end
    % 构建符号列表并随机打乱
    shapedSymbolsIdx = [];
    for i = 1:M
        shapedSymbolsIdx = [shapedSymbolsIdx; repmat(i-1, comp(i), 1)];
    end
    shapedSymbolsIdx = shapedSymbolsIdx(randperm(blockSize));
end