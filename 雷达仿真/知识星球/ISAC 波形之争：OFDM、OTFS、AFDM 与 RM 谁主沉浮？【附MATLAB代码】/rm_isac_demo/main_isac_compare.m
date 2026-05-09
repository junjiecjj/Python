% =======================================================================
% main_isac_compare.m
% -----------------------------------------------------------------------
% 单基站通信感知一体化演示：RM / OFDM / OTFS / AFDM
%
% 功能：
% 1) 基站发送一个通信帧，同时进行感知；
% 2) 用户端通过各自波形的解调/均衡进行通信接收；
% 3) 基站端缓存已知发射样本，针对双目标回波做交叉模糊函数(CAF)；
% 4) 可视化：
%    - 各波形的等效通信信道 |H_eff|
%    - 各波形的 CAF 图，并标注真实目标
%    - 各波形的均衡后星座图
%
% 说明：
% - 为了保证四种波形在同一套离散时延-多普勒物理信道下公平对比，
%   这里使用统一的块循环(block-circulant)离散模型；
% - OTFS 与 AFDM 采用“教学型统一线性算子实现”，便于直接对比，
%   而不是复现论文级的全脉冲成形/专用检测器。
% - 通信端统一采用符号域 LMMSE；感知端统一采用 CAF。
% =======================================================================
clc; clear; close all;rng(1,"twister")

%默认参数

params = struct();

% OTFS / RM / AFDM / OFDM 统一块长
params.M = 16;
params.N = 16;
params.L = params.M * params.N;

% 通信与感知的SNR
params.commSNRdB = 15;
params.radarSNRdB = 15;

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

% ------------------------------
% 1) 构造通信用户信道 / 双目标感知信道
% ------------------------------
H_comm = build_delay_doppler_channel(L,params.user.delays, params.user.dopplers, params.user.gains);

H_radar = build_delay_doppler_channel(L,params.target.delays, params.target.dopplers, params.target.gains);

% ------------------------------
% 2) 生成同一组QPSK信息符号
% ------------------------------
bitTx = randi([0, 1], 2*L, 1);
symTx = qpsk_mod(bitTx);

waveformList = {'RM', 'OFDM', 'OTFS', 'AFDM'};
results = struct();

fprintf('================ ISAC Demo Start ================\n');
fprintf('Block length L = %d, M = %d, N = %d\n', L, params.M, params.N);
fprintf('Comm SNR = %.1f dB, Radar SNR = %.1f dB\n', ...
    params.commSNRdB, params.radarSNRdB);
fprintf('=================================================\n\n');

for k = 1:numel(waveformList)
    waveformName = waveformList{k};

    % 3) 获取波形发送/接收算子
    op = get_waveform_operator(waveformName, params);

    % 4) 通信链路仿真
    commRes = simulate_comm_link(symTx, bitTx, op, H_comm, params.commSNRdB);

    % 5) 感知链路仿真（以同一个发射块做感知）
    radarRes = simulate_radar_link(commRes.txSamples, H_radar, ...
        params.radarSNRdB, params.delayGrid, params.dopplerGrid);

    % 6) 保存结果
    results(k).name = waveformName;
    results(k).operator = op;
    results(k).comm = commRes;
    results(k).radar = radarRes;

    % 7) 打印通信结果
    fprintf('%-5s | BER = %.4e | EVM = %.4f\n', ...
        waveformName, commRes.BER, commRes.EVM);

    % 8) 可视化：等效通信信道
    plot_effective_channel(commRes.Heff, waveformName);

    % 9) 可视化：CAF，并标注真实目标
    plot_caf_map(radarRes.CAF, params.delayGrid, params.dopplerGrid, ...
        params.target.delays, params.target.dopplers, waveformName);

    % 10) 可视化：均衡后星座图
    plot_constellation(commRes.symHat, symTx, waveformName, commRes.BER);
end

assignin('base', 'params_isac_demo', params);
assignin('base', 'results_isac_demo', results);

fprintf('\n结果结构体已写入工作区：results_isac_demo\n');
fprintf('参数结构体已写入工作区：params_isac_demo\n');
