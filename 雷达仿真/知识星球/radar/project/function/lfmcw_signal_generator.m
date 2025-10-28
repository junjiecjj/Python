function [channel_signal, ref_signal] = lfmcw_signal_generator(Fc, BW, T_chrip, T_idle, axis_t, tx_pos, rx_pos, target_info)
    % lfmcw_signal_generator  生成MIMO阵列接收信号混频后的中频信号
    % argument：
    %
    %  Fc     载波频率 (Hz)
    %  BW     带宽 (Hz)
    %  T_chrip   chirp持续时间 (s)
    %  T_idle  两个chirp之间的间隔时间 (s)
    %  axis_t  时间轴 (s)
    %  tx_pos  发射天线的位置,行数为3的二维矩阵，一列代表一个天线的位置 (xyz坐标)
    %  rx_pos  接收天线的位置,同上
    %  target_info 目标信息，每个目标用长度为6的数组表示，包含初始位置和速度
    %
    % return：
    %
    %  channel_signal 一个cell数组，每个元素对应一个通道的信号，大小为(numTx, numRx)
    %  ref_signal 第一个通道接受到的每一个信号源的信号
    %
    %  \remark
    %  - 函数计算不同目标在不同时间的位置、延迟、相位，然后生成信号。
    %  - 目标位置、延迟和相位是基于发射和接收天线的位置以及目标的速度计算得出的。
    % 建立坐标系时，按照右手螺旋原则，以雷达天线朝向为x轴正方向，，左侧为y轴正方向，上放为z轴正方向
    % 方位角：目标方向在xoy平面投影的角度，x轴正方向方位角为0°，y轴正方向方位角为90°
    % 俯仰角：目标方向和xoy平面的夹角，z轴正方向为90°俯仰角
    %   Example:
    %       BW = 250e6; %带宽
    %       Fc = 24e9; % 载波频率
    %       T_chrip = 420e-6; %  chirp 持续时间
    %       T_idle = 580e-6; % 两个chirp之间的间隔时间
    %       axis_t = 0:1/2.5e6:(T_chrip + T_idle) * 16; % 时间轴
    %       target_info = {[20, 10, 0, 10, 5, 0], [40, -20, 0, -20, 10, 0]};
    %       tx_pos = {[0, 1, 1], [0, 3, 1]};
    %       rx_pos = {[0, 0, 0], [0, 1, 0], [0, 2, 0]};
    %       [channel_signal, ref_signal] = lfmcw_signal_generator(Fc, BW, T_chrip, T_idle, axis_t, tx_pos, rx_pos, target_info);
    %       figure(1)
    %       subplot(211);
    %       plot(real(ref_signal(1, :)));
    %       subplot(212);
    %       plot(real(ref_signal(2, :)));
    arguments
        Fc double
        BW double
        T_chrip double
        T_idle double
        axis_t (1, :) double
        tx_pos (3, :) double
        rx_pos (3, :) double
        target_info (6, :) double
    end
    % 参数计算
    c = physconst('LightSpeed'); %光速
    f_slope = BW / T_chrip; %调频斜率

    numTx = size(tx_pos, 2);
    numRx = size(rx_pos, 2);
    numTargets = size(target_info, 2);

    % 计算不同时间的目标位置信息
    targets_pos = cell(size(target_info, 2), 1);
    for i = 1:size(target_info, 2)
        r0 = target_info(1:3, i);
        v = target_info(4:6, i);
        targets_pos{i} = r0 + v * axis_t; % 目标位置，3×N矩阵，储存N个时刻的xyz坐标
    end

    % 计算不同通道在不同时间的延迟
    targets_delay = cell(numTx, numRx);
    for i = 1:numTx
        for j = 1:numRx
            targets_delay{i, j} = zeros(numTargets, length(axis_t));
            for k = 1:numTargets
                targets_delay{i, j}(k, :) = (vecnorm(targets_pos{k} - tx_pos(:, i), 2, 1) + vecnorm(targets_pos{k} - rx_pos(:, j), 2, 1)) / c;
            end
        end
    end

    % 计算每个通道中每个目标的相位
    targets_phase = cell(numTx, numRx);
    for i = 1:numTx
        for j = 1:numRx
            targets_phase{i, j} = clac_Phase_tx(repmat(axis_t, numTargets, 1) - targets_delay{i, j}, Fc, T_chrip, T_idle, f_slope);

        end
    end
    tx_phase = clac_Phase_tx(axis_t, Fc, T_chrip, T_idle, f_slope);

    % 计算每个通道的接收信号
    channel_signal = cell(numTx, numRx);
    for i = 1:numTx
        for j = 1:numRx
            channel_signal{i, j} = sum(exp(1j * (targets_phase{i, j} - tx_phase)), 1);
        end
    end
    ref_signal = exp(1j * (targets_phase{1, 1} - tx_phase));

end

%% 计算相位函数
function signal = clac_Phase_tx(axis_t, fc, T_chrip, T_idle, slope)
    t = mod(axis_t, T_chrip + T_idle);
    signal = zeros(size(t));
    signal(t < T_chrip) = 2 * pi * (t(t < T_chrip) .* (fc + 0.5 * slope * t(t < T_chrip)));
    signal(t >= T_chrip) = 2 * pi * fc .* t(t >= T_chrip);
end
