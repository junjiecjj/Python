

function [Pcapon, angle_est, peaks] = Capon(Rxx, Thetalst, N)
    % Capon 算法（MVDR）用于 DOA 估计
    % 输入：
    %   Rxx : N×N 接收信号协方差矩阵（Hermitian）
    %   K   : 信源个数（Capon 算法中未使用，保留仅为接口一致）
    %   N   : 阵元个数
    % 输出：
    %   Thetalst  : 角度网格（度）
    %   Pcapon    : 空间谱（dB）
    %   angle_est : 估计的波达方向（度）
    %   peaks     : 峰值索引（MATLAB 1‑based）

    % 角度扫描范围与步长
    Pcapon = zeros(size(Thetalst));

    % 导向矢量的索引 d = [0, 1, ..., N-1]'（列向量）
    d = (0:N-1)';

    % 计算协方差矩阵的逆
    % Rxx_inv = inv(Rxx);                          % 注意：Rxx 应满秩

    for i = 1:length(Thetalst)
        ang = Thetalst(i);
        a = exp(-1j * pi * d * sind(ang));        % 导向矢量（列向量）
        denom = real(a' / Rxx * a);          % 分母取实部
        Pcapon(i) = 1 / denom;                   % Capon 谱值
    end

    % 归一化并转换到 dB
    Pcapon = abs(Pcapon) / max(abs(Pcapon));
    Pcapon = 10 * log10(Pcapon);

    % 峰值检测（需要 Signal Processing Toolbox）
    % 高度阈值 -2 dB，最小间隔 10 个采样点（对应 5°）
    [~, locs] = findpeaks(Pcapon, 'MinPeakHeight', -2, 'MinPeakDistance', 10);
    peaks = locs;                                % 1‑based 索引
    angle_est = Thetalst(peaks);
end