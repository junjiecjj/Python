#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:23:29 2025

@author: jack
"""

# https://www.cnblogs.com/gjblog/p/13352746.html

# https://www.cnblogs.com/gjblog/p/13363995.html

# https://github.com/loveybos/CommunicationSys





%%  https://www.cnblogs.com/gjblog/p/13352746.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Version 3.0
%%% 16QAM调制(官方函数)
%%% IFFT(官方函数)
%%% 添加循环前缀(单径AWGN中CP好像没啥用？？)
%%% AWGN信道
%%% 去除循环前缀
%%% FFT(官方函数)
%%% 16QAM解调(官方函数)
%%% BER分析
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;
%% 基带数字信号及一些初始化
NFRAME = 3; % 每一帧的OFDM符号数
NFFT = 64; % 每帧FFT长度
NCP = 16; % 循环前缀长度
NSYM = NFFT + NCP; % OFDM符号长度
M = 16; K = 4; % M:调制阶数，K:log2(M)

EbN0 = 0:1:20; % 设出比特信噪比(dB)
snr = EbN0 + 10 * log10(K); % 由他俩关系推出(dB)
BER(1 : length(EbN0)) = 0; % 初始化误码率

file_name=['OFDM_BER_NCP' num2str(NCP) '.dat'];
fid=fopen(file_name, 'w+');

X = randi([0,15],1,NFFT * NFRAME); % 生成数字基带信号
%%
for i = 1 : length(EbN0) % 对于每一种比特信噪比，计算该通信环境下的误码率
    %% 16QAM调制(官方函数)
    X_mod = qammod(X,M,'gray') / (sqrt(10)); % 16QAM调制，格雷码星座图，并归一化
    %% IFFT与循环前缀添加
    x(1 : NFFT * NFRAME) = 0; % 预分配x数组
    xt(1 : (NFFT + NCP) * NFRAME) = 0; % 预分配x_t数组

    len_a = 1 : NFFT; % 处理的X位置
    len_b = 1 : (NFFT + NCP); % 处理的X位置(加上CP的长度)
    len_c = 1 : NCP;
    len_left = 1 : (NFFT / 2); len_right = (NFFT / 2 + 1) : NFFT; % 每一符号分为左右两边？？

    for frame = 1 : NFRAME % 对于每个OFDM符号都要翻转和IFFT
        x(len_a) = ifft([X_mod(len_right), X_mod(len_left)]); % 左右翻转再ifft
        xt(len_b) = [x(len_c + NFFT - NCP), x(len_a)]; % 添加CP后的信号数组

        len_a = len_a + NFFT; % 更新找到下一个符号起点位置
        len_b = len_b + NFFT + NCP; % 更新找到下一个符号起点位置(CP开头)
        len_c = len_c + NFFT;
        len_left = len_left + NFFT; len_right = len_right + NFFT;
    end
    %% 由snr计算噪声幅度并加噪
%      P_s = xt * xt' / NSYM / NFRAME; % 计算信号功率
%       delta1 = sqrt(10 .^ (-snr(i) / 10) * P_s / 2); % 计算噪声标准差
%       yr = delta1 * (randn(size(xt)) + 1i * randn(size(xt)));
    yr = awgn(xt,snr(i),'measured'); % 官方函数直接加噪
    %% 去除循环前缀并且FFT
    y(1 : NFFT * NFRAME) = 0; % 预分配y数组
    Y(1 : NFFT * NFRAME) = 0; % 预分配Y数组

    len_a = 1 : NFFT; % 处理的y位置
    len_b = 1 : NFFT; % 处理的y位置
    len_left = 1 : (NFFT / 2); len_right = (NFFT / 2 + 1) : NFFT; % 每一符号分为左右两边

    for frame = 1 : NFRAME % 对于每个OFDM符号先去CP，再FFT再翻转
        y(len_a) = yr(len_b + NCP); % 去掉CP

        Y(len_a) = fft(y(len_a)); % 先fft再翻转
        Y(len_a) = [Y(len_right), Y(len_left)];

        len_a = len_a + NFFT;
        len_b = len_b + NFFT + NCP;
        len_left = len_left + NFFT; len_right = len_right + NFFT;
    end
    %% 16QAM解调(官方函数)
    Yr = qamdemod(Y * sqrt(10),M,'gray');
    %% BER计算
    Neb = sum(sum(de2bi(Yr,K) ~= de2bi(X,K))); % 转为2进制，计算具体有几bit错误
    Ntb = NFFT * NFRAME * K;  %仿真的总比特数
    BER(i) = Neb / Ntb;

    fprintf('EbN0 = %d[dB], BER = %d / %d = %.3f\n', EbN0(i),Neb,Ntb,BER(i))
    fprintf(fid, '%d %5.3f\n', EbN0(i),BER(i));
end
%% BER作图分析
fclose(fid);
disp('Simulation is finished');
plot_ber(file_name,K);

%% https://www.cnblogs.com/gjblog/p/13363995.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Version 3.1
%%% 16QAM调制(官方函数)
%%% IFFT(官方函数)
%%% 添加循环前缀
%%% 经过多径瑞利衰减信道
%%% 添加AWGN
%%% 去除循环前缀
%%% FFT(官方函数)
%%% 16QAM解调(官方函数)
%%% BER分析
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;
%% 基带数字信号及一些初始化
NFRAME = 3;      % 每一帧的OFDM符号数
NFFT = 64;         % 每帧FFT长度
NCP = 16;          % 循环前缀长度
NSYM = NFFT + NCP; % OFDM符号长度
M = 16; K = 4;     % M:调制阶数，K:log2(M)

P_hdB = [0 -8 -17 -21 -25];     % 各信道功率特性(dB)
D_h = [0 3 5 6 8];              % 各信道延迟(采样点)
P_h = 10 .^ (P_hdB / 10);       % 各信道功率特性
NH = length(P_hdB);             % 多径信道个数
LH = D_h(end)+1;                % 信道长度(延迟过后)

EbN0 = 0:1:20;              % 设出比特信噪比(dB)
snr = EbN0 + 10 * log10(K); % 由比特信噪比计算出snr(dB)
BER(1 : length(EbN0)) = 0;  % 初始化误码率

file_name=['OFDM_BER_NCP' num2str(NCP) '.dat'];
fid=fopen(file_name, 'w+');

X = randi([0,15],1,NFFT * NFRAME); % 生成基带数字信号
%%
for i = 1 : length(EbN0) % 对于每一种比特信噪比，计算该通信环境下的误码率

    randn('state',0); % 很重要！！保持信道参数在每一次仿真中都不变
    rand('state',0);

    %% 16QAM调制(官方函数)
    X_mod = qammod(X,M,'gray') / (sqrt(10)); % 16QAM调制，格雷码星座图，并归一化
    %% IFFT与循环前缀添加
    x(1 : NFFT * NFRAME) = 0; % 预分配x数组
    xt(1 : (NFFT + NCP) * NFRAME) = 0; % 预分配xt数组

    len_a = 1 : NFFT; % 处理的X位置
    len_b = 1 : (NFFT + NCP); % 处理的X位置(加上CP的长度)
    len_c = 1 : NCP;
    len_left = 1 : (NFFT / 2); len_right = (NFFT / 2 + 1) : NFFT; % 每一符号分为左右两边？？

    for frame = 1 : NFRAME % 对于每个OFDM符号都要翻转和IFFT
        x(len_a) = ifft([X_mod(len_right), X_mod(len_left)]); % 左右翻转再ifft
        xt(len_b) = [x(len_c + NFFT - NCP), x(len_a)]; % 添加CP后的信号数组

        len_a = len_a + NFFT; % 更新找到下一个符号起点位置
        len_b = len_b + NFFT + NCP; % 更新找到下一个符号起点位置(CP开头)
        len_c = len_c + NFFT;
        len_left = len_left + NFFT; len_right = len_right + NFFT;
    end
    %% 经过多径瑞利衰减信道
    A_h = (randn(1,NH) + 1i * randn(1,NH)) .* sqrt(P_h / 2); % 生成瑞利随机变量
    h = zeros(1,LH); % 初始化信道冲激响应模型
    h(D_h + 1) = A_h; % 信道冲激响应(同时体现出衰减系数和信道时延)
    xt1 = conv(xt,h); % 卷积，输出通过该信道的信号
    %% 由snr计算噪声幅度并加噪
    xt2 = xt1(1 : NSYM * NFRAME); % 只取卷积过后属于OFDM符号的部分
    P_s = xt2 * xt2' ./ NSYM ./ NFRAME; % 计算信号功率

    A_n = sqrt(10 .^ (-snr(i) / 10) * P_s / 2); % 计算噪声标准差
    yr = xt1 + A_n * (randn(size(xt1)) + 1i * randn(size(xt1))); % 根据噪声标准差添加噪声
    %% 去除循环前缀并且FFT
    y(1 : NFFT * NFRAME) = 0; % 预分配y数组
    Y(1 : NFFT * NFRAME) = 0; % 预分配Y数组

    len_a = 1 : NFFT; % 处理的y位置
    len_b = 1 : NFFT; % 处理的y位置
    len_left = 1 : (NFFT / 2); len_right = (NFFT / 2 + 1) : NFFT; % 每一符号分为左右两边

     H= fft([h zeros(1,NFFT-LH)]); % 信道频率响应
     H_shift(len_a)= [H(len_right) H(len_left)];

    for frame = 1 : NFRAME % 对于每个OFDM符号先去CP，再FFT再翻转
        y(len_a) = yr(len_b + NCP); % 去掉CP

        Y(len_a) = fft(y(len_a)); % 先fft再翻转
        Y(len_a) = [Y(len_right), Y(len_left)] ./ H_shift; % 信道均衡（直接除以理想信道，没有进行信道估计！）

        len_a = len_a + NFFT;
        len_b = len_b + NFFT + NCP;
        len_left = len_left + NFFT; len_right = len_right + NFFT;
    end
    %% 16QAM解调(官方函数)
    Yr = qamdemod(Y * sqrt(10),M,'gray');
    %% BER计算(多次迭代算均值会更准确)
    Neb = sum(sum(de2bi(Yr,K) ~= de2bi(X,K))); % 转为2进制，计算具体有几bit错误
    Ntb = NFFT * NFRAME * K;  %[Ber,Neb,Ntb]=ber(bit_Rx,bit,Nbps);
    BER(i) = Neb / Ntb;
    fprintf('EbN0 = %3d[dB], BER = %4d / %8d = %11.3e\n', EbN0(i),Neb,Ntb,BER(i))
    fprintf(fid, '%d %11.3e\n', EbN0(i),BER(i));
end
%% BER作图分析
fclose(fid);
disp('Simulation is finished');
plot_ber(file_name,K);






