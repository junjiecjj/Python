function varargout = lfmcw_radar_data_cube_generator(freq_carry, bandwidth, T_chrip, T_idle, T_nop, freq_sampling, numSampling, numChrip, numFrame, tx_pos, rx_pos, target_info)
    %
    % 生成MIMO雷达数据立方体，模拟中频信号经过混频后的接收信号
    %
    % 输入参数：
    %   Fc: 载波频率 (Hz)
    %   BW: 带宽 (Hz)
    %   T_chrip: chirp持续时间 (s)
    %   T_idle: 两个chirp之间的间隔时间 (s)
    %   T_nop: 空闲时间 (s)
    %   freq_sampling: 采样频率 (Hz)
    %   numSampling: chirp内采样点数
    %   numChrip: chirp序列数量
    %   numFrame: 帧数
    %   tx_pos: 发射天线位置 (3xnumTx矩阵，xyz坐标)
    %   rx_pos: 接收天线位置 (3xnumRx矩阵，xyz坐标)
    %   target_info: 目标信息 (6xn矩阵，每行表示一个目标的初始位置和速度)
    %
    % 输出参数：
    %   radar_data_cube: 数据立方体 (numFrame x numRx*numTx x numChrip x numSampling)
    %   res_range: 距离分辨率 (m)
    %   res_velocity: 速度分辨率 (m/s)
    %
    % 注释：
    %   - 根据目标信息、天线位置和时间轴计算目标位置、延迟和相位
    %   - 基于右手螺旋坐标系，x轴正向，y轴左侧，z轴上方
    %   - 方位角：目标方向在xoy平面的投影角度
    %   - 俯仰角：目标方向与xoy平面的夹角
    %
    % EXAMPLE:
    %   bandWidth = 250e6; %带宽
    %   fc = 24e9; % 载波频率
    %   T_chrip = 420e-6; %  chirp 持续时间
    %   T_idle = 580e-6; % 两个chirp之间的间隔时间
    %   Fs = 6e5;
    %   numSampling = 256; % # ADC采样点数/chrip
    %   numChrip = 32; % # chrip/frame
    %   numFrame = 16; % frame数量
    %   numTx = 1; % 发射天线数量
    %   numRx = 8; % 接收天线数量
    %   d_rx = lambda / 2; % 接收天线之间的距离，单位米
    %   d_tx = 4 * d_rx; % 发射天线之间的距离，单位米
    %   % 目标初始化信息，每个目标用长度为6的列向量保存信息，前三位代表起始坐标，后三位代表速度。
    %   target_info = [[10 * cosd(10), 10 * sind(10), 0, 0, 0, 0].', ...
    %                       [80 * cosd(-45), 80 * sind(-45), 0, 10 * cosd(-45), 10 * sind(-45), 0].'];
    %   % 发射天线的位置，xyz坐标系
    %   tx_pos = [zeros(1, numTx); linspace(-0.5, 0.5, numTx) * d_tx * (numTx - 1); zeros(1, numTx)];
    %   % 接受天线的位置
    %   rx_pos = [zeros(1, numRx); linspace(-0.5, 0.5, numRx) * d_rx * (numRx - 1); zeros(1, numRx)];
    %   radar_data_cube = lfmcw_radar_data_cube_generator(fc, bandWidth, T_chrip, T_idle, 0.4, Fs , numSampling, numChrip, numFrame, tx_pos, rx_pos, target_info);

    arguments
        freq_carry double
        bandwidth double
        T_chrip double
        T_idle double
        T_nop double
        freq_sampling double
        numSampling double
        numChrip double
        numFrame double
        tx_pos (3, :) double
        rx_pos (3, :) double
        target_info (6, :) double
    end

    % 参数计算
    c = physconst('LightSpeed'); %光速
    f_slope = bandwidth / T_chrip; %调频斜率

    numTx = size(tx_pos, 2);
    numRx = size(rx_pos, 2);
    numTargets = size(target_info, 2);

    t = T_chrip - (numSampling - 1) / freq_sampling;

    if (t < 0)
        error("chrip时长需要大于 (numSampling - 1) / freq_sampling")
    end
    axis_4 = (0:numSampling - 1) ./ freq_sampling +t / 2;
    axis_3 = (0:numChrip - 1) .* (T_chrip + T_idle);
    axis_2 = (0:numFrame -1) .* (T_nop +numChrip * (T_chrip + T_idle));
    axis_t = kron(axis_3, ones(1, numSampling)) + repmat(axis_4, 1, numChrip);
    axis_t = kron(axis_2, ones(size(axis_t))) + repmat(axis_t, 1, numFrame);

    % 计算不同时间的目标位置信息
    targets_pos = zeros([numTargets, 3, size(axis_t, 2)]);
    for i = 1:size(target_info, 2)
        r0 = target_info(1:3, i);
        v = target_info(4:6, i);
        targets_pos(i, :, :) = r0 + v * axis_t; % 目标位置，3×N矩阵，储存N个时刻的xyz坐标
    end

    % 计算不同通道在不同时间的延迟
    targets_delay = zeros([numRx * numTx, numTargets, size(axis_t, 2)]);
    for i = 1:numTx
        for j = 1:numRx
            for k = 1:numTargets
                targets_delay((i - 1) * numRx + j, k, :) = (vecnorm(squeeze(targets_pos(k, :, :)) - tx_pos(:, i), 2, 1) + vecnorm(squeeze(targets_pos(k, :, :)) - rx_pos(:, j), 2, 1)) / c;
            end
        end
    end

    % 计算每个通道中每个目标的相位
    targets_phase = zeros([numRx * numTx, numTargets, size(axis_t, 2)]);
    for i = 1:numTx
        for j = 1:numRx
            targets_phase((i - 1) * numRx + j, :, :) = clac_Phase_tx(repmat(axis_t, numTargets, 1) - squeeze(targets_delay((i - 1) * numRx + j, :, :)), freq_carry, T_chrip, T_idle, T_nop, numChrip, f_slope);

        end
    end
    tx_phase = clac_Phase_tx(axis_t, freq_carry, T_chrip, T_idle, T_nop, numChrip, f_slope);

    % 计算每个通道的接收信号
    channel_signal = zeros([size(axis_t, 2), numRx * numTx]);
    for i = 1:numTx
        for j = 1:numRx
            channel_signal(:, (i - 1) * numRx + j) = sum(exp(1j * (squeeze(targets_phase((i - 1) * numRx + j, :, :)) - tx_phase)), 1);
        end
    end
    % ref_signal = exp(1j * (targets_phase(1, 1, :, :) - tx_phase));

    radar_data_cube = reshape(channel_signal, [numSampling, numChrip, numFrame, numRx * numTx]);
    radar_data_cube = permute(radar_data_cube, [3, 4, 2, 1]);
    useful_bandwidth = (numSampling - 1) / freq_sampling * f_slope;
    res_range = c / (2 * useful_bandwidth);
    res_velocity = physconst('LightSpeed') / (2 * freq_carry * (T_chrip + T_idle) * numChrip);
    % 确定输出的数量
    nargoutchk(0, 3); % 检查输出参数的数量，最多支持3个

    % 分配输出参数
    switch nargout
        case 1
            varargout{1} = radar_data_cube;
        case 2
            varargout{1} = radar_data_cube;
            varargout{2} = res_range;
        case 3
            varargout{1} = radar_data_cube;
            varargout{2} = res_range;
            varargout{3} = res_velocity;
        otherwise
            error('Too many output arguments');
    end

end

%% 计算相位函数
function signal = clac_Phase_tx(axis_t, fc, T_chrip, T_idle, T_nop, numChrip, slope)
    t = mod(axis_t, (T_chrip + T_idle) * numChrip + T_nop);
    t = mod(t, T_chrip + T_idle);
    signal = 2 * pi * (t .* (fc + 0.5 * slope * t));
end
