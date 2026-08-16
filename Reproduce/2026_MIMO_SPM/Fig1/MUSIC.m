function [Pmusic, angle_est, peaks] = MUSIC(Ryy,Thetalst, K, N)
    % MUSIC 算法用于 DOA 估计
    % 输入：
    %   Rxx : N×N 接收信号协方差矩阵（Hermitian 矩阵）
    %   K   : 信源个数
    %   N   : 阵列天线个数
    % 输出：
    %   Thetalst  : 角度网格（度）
    %   Pmusic    : 空间谱（dB）
    %   angle_est : 估计的波达方向（度）
    %   peaks     :  

    % 特征值分解
    [eigvector, ~] = eig(Ryy);          % eigvector 的列是特征向量

    % 提取噪声子空间（后 N-K 列）
    Un = eigvector(:, 1:K+1);                  % MATLAB 索引从 1 开始

    % 计算噪声子空间的投影矩阵 Un * Un^H
    UnUnH = Un * Un';                          % 共轭转置用 ' 实现

    % 角度扫描范围与步长
    Pmusic = zeros(size(Thetalst));

    % 计算每个角度对应的 MUSIC 谱值
    for i = 1:length(Thetalst)
        ang = Thetalst(i);
        a = exp(1j * pi * (0:N-1)' * sind(ang)); % 导向矢量（列向量）
        denom = a' * UnUnH * a;                  % 分母（标量）
        Pmusic(i) = 1 / abs(denom);              % 谱值
    end

    % 归一化（线性幅度）
    Pmusic = abs(Pmusic) / max(abs(Pmusic));
    [~, locs] = findpeaks(Pmusic, 'MinPeakHeight', 0.1*max(Pmusic), 'MinPeakDistance', 5);

    % 转换为 dB
    Pmusic = 10 * log10(Pmusic);

    % 峰值检测,高度阈值 -10 dB，最小间隔 10 个采样点（对应 5°）
    % [~, locs] = findpeaks(Pmusic, 'MinPeakHeight', -10, 'MinPeakDistance', 10);
    % [peaks, locs] = findpeaks(Pmusic, Thetalst, 'MinPeakHeight', max(Pmusic) - 30, 'MinPeakDistance', 5);
    peaks = Pmusic(locs);                            
    angle_est = Thetalst(locs);               % 对应的角度估计
end