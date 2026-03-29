%% 256QAM 概率整形仿真

clear all;
clc;
close all;

%% 参数设置
M = 256;                   % 256QAM 调制阶数
k = log2(M);              % 每个符号比特数
Nsym_total = 1e5;         % 总符号数（用于 BER 仿真）
blockSize = 1e4;          % 每块符号数（防止一次生成过大数组）
nBlocks = ceil(Nsym_total / blockSize);

SNR_dB = 0:2:40;          % 信噪比范围（dB）
maxSNR = max(SNR_dB);

%% 生成标准 64QAM 星座图（均匀概率，单位平均功率）
uniformConstellation = qammod(0:M-1, M, 'UnitAveragePower', true, 'PlotConstellation', false);

% 使用初始 lambda=1 进行展示
lambda_display = 1;
P_unorm_display = exp(-lambda_display * abs(uniformConstellation).^2);
P_shape_display = P_unorm_display / sum(P_unorm_display);
E_shape_display = sum(P_shape_display .* abs(uniformConstellation).^2);
shapedConstellation_display = uniformConstellation / sqrt(E_shape_display);

%% 3D 柱状图显示符号分布
N_forHist = 1e6;
% (1) 均匀符号
uniformSymbolsIdx_hist = randi([0, M-1], N_forHist, 1);
txUniform_hist = uniformConstellation(uniformSymbolsIdx_hist+1);
% (2) 概率整形符号（使用 CCDM_DM）
shapedSymbolsIdx_hist = CCDM_DM(P_shape_display, N_forHist, M);
txShaped_hist = shapedConstellation_display(shapedSymbolsIdx_hist+1);

% 计算均匀64QAM的最大范围
realMax_uniform = max(abs(real(txUniform_hist)));
imagMax_uniform = max(abs(imag(txUniform_hist)));

% 计算概率整形64QAM的最大范围
realMax_shaped = max(abs(real(txShaped_hist)));
imagMax_shaped = max(abs(imag(txShaped_hist)));

% 生成不同的边界
numBins = 16;  
xEdges_uniform = linspace(-realMax_uniform, realMax_uniform, numBins+1);
yEdges_uniform = linspace(-imagMax_uniform, imagMax_uniform, numBins+1);
xEdges_shaped = linspace(-realMax_shaped, realMax_shaped, numBins+1);
yEdges_shaped = linspace(-imagMax_shaped, imagMax_shaped, numBins+1);

% 分别计算直方图
countsUniform = histcounts2(real(txUniform_hist), imag(txUniform_hist), xEdges_uniform, yEdges_uniform);
countsShaped  = histcounts2(real(txShaped_hist), imag(txShaped_hist), xEdges_shaped, yEdges_shaped);
countsUniform = countsUniform / sum(countsUniform(:));
countsShaped  = countsShaped  / sum(countsShaped(:));

% 绘制均匀 64QAM 符号分布
figure;
bar3(countsUniform);
title('均匀 256QAM 实际符号分布');
xlabel('Real bin');
ylabel('Imag bin');
zlabel('概率');
colormap turbo;
axis tight;

% 绘制概率整形 64QAM 符号分布
figure;
bar3(countsShaped);
title('CCDM 概率整形 256QAM 实际符号分布');
xlabel('Real bin');
ylabel('Imag bin');
zlabel('概率');
colormap turbo;
axis tight;


%% 仿真：块处理方式统计不同 SNR 下的误码率（BER）
BER_uniform = zeros(length(SNR_dB), 1);
BER_shaped  = zeros(length(SNR_dB), 1);

for snrIdx = 1:length(SNR_dB)
    snr = SNR_dB(snrIdx);
    noiseVariance = 1/(10^(snr/10));

    % 自适应整形参数：SNR 较低时采用较高 lambda（减小外圈符号概率）
    lambda_current = 1 + 0.05*(maxSNR - snr);
    % 重新计算概率整形参数
    P_unorm = exp(-lambda_current * abs(uniformConstellation).^2);
    P_shape = P_unorm / sum(P_unorm);
    E_shape = sum(P_shape .* abs(uniformConstellation).^2);
    shapedConstellation = uniformConstellation / sqrt(E_shape);

    errUniform_total = 0;
    errShaped_total = 0;
    totalBits = 0;

    for blk = 1:nBlocks
        currentBlockSize = min(blockSize, Nsym_total - (blk-1)*blockSize);

        % 生成均匀符号索引
        uniformSymbolsIdx = randi([0, M-1], currentBlockSize, 1);
        % CCDM 映射：固定组成 DM
        shapedSymbolsIdx = CCDM_DM(P_shape, currentBlockSize, M);

        % 映射为星座点
        txUniform = uniformConstellation(uniformSymbolsIdx+1).';
        txUniform = txUniform(:);
        txShaped  = shapedConstellation(shapedSymbolsIdx+1).';
        txShaped = txShaped(:);

        % 生成 AWGN 噪声
        noiseUniform = sqrt(noiseVariance/2) * (randn(currentBlockSize, 1) + 1i*randn(currentBlockSize, 1));
        noiseShaped  = sqrt(noiseVariance/2) * (randn(currentBlockSize, 1) + 1i*randn(currentBlockSize, 1));

        % 信道传输
        rxUniform = txUniform + noiseUniform;
        rxShaped  = txShaped + noiseShaped;

        if snr==28 && blk==10
            scatterplot(rxUniform);
            scatterplot(rxShaped);
        end

        % 判决检测
        rxUniformIdx = qamdemod(rxUniform, M, 'UnitAveragePower', true, 'PlotConstellation', false);
        rxShapedIdx = zeros(currentBlockSize, 1);
        for n = 1:currentBlockSize
            distances = abs(rxShaped(n) - shapedConstellation).^2;
            [~, minIndex] = min(distances);
            rxShapedIdx(n) = minIndex - 1;
        end

        % 转换为比特流（left-msb 格式）
        txUniformBits = de2bi(uniformSymbolsIdx, k, 'left-msb');
        rxUniformBits = de2bi(rxUniformIdx, k, 'left-msb');
        txShapedBits  = de2bi(shapedSymbolsIdx, k, 'left-msb');
        rxShapedBits  = de2bi(rxShapedIdx, k, 'left-msb');

        errUniform_total = errUniform_total + sum(sum(txUniformBits ~= rxUniformBits));
        errShaped_total  = errShaped_total + sum(sum(txShapedBits ~= rxShapedBits));
        totalBits = totalBits + currentBlockSize * k;
    end

    BER_uniform(snrIdx) = errUniform_total / totalBits;
    BER_shaped(snrIdx)  = errShaped_total / totalBits;

    fprintf('SNR = %d dB: 均匀 BER = %e, CCDM 概率整形 BER = %e\n', snr, BER_uniform(snrIdx), BER_shaped(snrIdx));
end

%% 绘制 BER 曲线
figure;
semilogy(SNR_dB, BER_uniform, 'b-o', 'LineWidth', 2);
hold on;
semilogy(SNR_dB, BER_shaped, 'r-s', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('误码率 (BER)');
title('均匀 vs CCDM 概率整形 256QAM 的 BER 对比');
legend('均匀 256QAM', 'CCDM 概率整形 256QAM', 'Location', 'southwest');

%% --- CCDM_DM 函数 ---
function shapedSymbolsIdx = CCDM_DM(P_shape, blockSize, M)
    % CCDM_DM - Constant Composition Distribution Matching 的简单实现
    % 输入:
    %   P_shape   : 长度为 M 的目标概率分布（如 Maxwell–Boltzmann 分布）
    %   blockSize : 块长度（符号数）
    %   M         : 调制阶数
    % 输出:
    %   shapedSymbolsIdx : 长度为 blockSize 的符号索引（0 到 M-1）
    
    % 根据目标概率计算每个星座点应出现的次数（四舍五入）
    comp = round(P_shape * blockSize);
    % 调整使得总数正好等于 blockSize
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
    % 构造固定组成的符号序列
    shapedSymbolsIdx = [];
    for i = 1:M
        shapedSymbolsIdx = [shapedSymbolsIdx; repmat(i-1, comp(i), 1)];
    end
    % 随机打乱顺序
    shapedSymbolsIdx = shapedSymbolsIdx(randperm(blockSize));
end
