function [echo_data, range_compressed, isar_image] = module_isar_imaging(config)
%% module_isar_imaging - ISAR成像模块（供 Main_System.m 调用）
%
% 输入：
%   config - 参数结构体，需包含以下字段：
%            config.fc            载波频率 (Hz)，如 28e9
%            config.B             信号带宽 (Hz)，如 400e6
%            config.PRF           脉冲重复频率 (Hz)，如 1000
%            config.T_obs         观测时间 (s)，如 2
%            config.target_range  参考距离 R0 (m)，如 1000
%            config.rotation_rate 旋转角速度 ω (rad/s)，如 0.5
%
% 输出：
%   echo_data        - 原始回波矩阵 [num_samples × num_pulses]，复数
%   range_compressed - 距离压缩后矩阵（同尺寸），复数
%   isar_image       - ISAR成像结果（同尺寸），复数

    %% 1. 派生参数（由 config 计算得到）
    c      = 3e8;                 % 光速 (m/s)
    Tp     = 1e-6;                % 脉冲宽度（固定 1μs，未放入 config）
    fs     = 2 * config.B;        % 采样率 = 2B（奈奎斯特定理，保证不混叠）
    Kr     = config.B / Tp;       % 调频率 Kr = B/Tp，chirp 频率变化速率

    %% 2. 无人机目标模型（4个散射点，十字形）
    % 每行 [x, y, z]（单位：米），z 轴在2D成像中不参与计算
    % 对应理论笔记 §2.3 散射中心坐标
    target_points = [
         0,    0,    0;    % 中心点
         0.3,  0,    0;    % 右臂
        -0.3,  0,    0;    % 左臂
         0,    0.25, 0     % 前臂
    ];
    num_points = size(target_points, 1);  % = 4

    %% 3. 时间轴
    % 慢时间：每个脉冲的发射时刻，步长 = 1/PRF
    slow_time   = 0:1/config.PRF:config.T_obs;
    num_pulses  = length(slow_time);   % PRF×T_obs + 1 = 2001

    % 快时间：单个脉冲内的采样时刻，步长 = 1/fs = 1.25ns
    fast_time   = 0:1/fs:Tp;
    num_samples = length(fast_time);   % 2B×Tp + 1 = 801

    %% 4. 生成回波数据
    % 对应理论笔记 §2.4 回波信号生成
    echo_data = zeros(num_samples, num_pulses);  % 预分配：行=距离维，列=方位维
    R0 = config.target_range;                   % 参考距离（基站到无人机初始距离）

    for p = 1:num_pulses
        % 第 p 个脉冲时刻，无人机旋转角度：θ(t_m) = ω × t_m
        theta = config.rotation_rate * slow_time(p);

        for k = 1:num_points
            % 旋转变换（对应理论公式）：
            %   x_k' = x_k·cosθ − y_k·sinθ  （方位维，不影响径向距离，无需使用）
            %   y_k' = x_k·sinθ + y_k·cosθ  （距离维，决定瞬时距离 R）
            y_rot = target_points(k,1)*sin(theta) + target_points(k,2)*cos(theta);

            % 瞬时距离：R_k = R0 + y_k'
            % y 方向为雷达视线方向（LOS），仅 y_rot 改变径向距离
            R = R0 + y_rot;

            % 相对时延：τ_k = 2(R_k − R0)/c
            % 【关键】用相对时延而非绝对时延 2R/c：
            %   绝对时延 ≈ 6.67μs >> 快时间窗口 Tp=1μs，信号会全部清零
            %   相对时延 ≈ ±2ns << 1μs，信号完整落在采样窗口内
            tau = 2*(R - R0)/c;

            % 散射点回波：s_k(t,t_m) = exp(jπKr(t−τ_k)²) × exp(−j4πfc·R/c)
            %
            % 第一项：LFM chirp（快时间项），τ_k 决定距离维峰值位置
            %   (fast_time - tau)：行向量（1×801）减标量，结果仍为行向量
            %
            % 第二项：载波相位（慢时间项），R 随旋转变化产生多普勒频移
            %   4π = 往返传播两次相位累积（发射+接收）
            %   此项随 t_m 变化（因 R 变化），FFT 后在方位维分离散射点
            sig = exp(1j*pi*Kr*(fast_time - tau).^2) .* ...
                  exp(-1j*4*pi*config.fc*R/c);

            % sig 为行向量（1×801），转置为列向量（801×1）后累加
            echo_data(:, p) = echo_data(:, p) + sig';
        end
    end

    % 添加高斯白噪声，SNR = 10dB
    SNR_dB = 10;
    try
        % Communications Toolbox 方法：'measured' 以实测信号功率为基准
        echo_data = awgn(echo_data, SNR_dB, 'measured');
    catch
        % 手动添加：由 SNR 反算噪声功率，再生成复高斯随机噪声
        signal_power = mean(abs(echo_data(:)).^2);
        noise_power  = signal_power / (10^(SNR_dB/10));
        % 实部、虚部各贡献 noise_power/2（等功率分配），标准差 = sqrt(noise_power/2)
        noise = sqrt(noise_power/2) * (randn(size(echo_data)) + 1j*randn(size(echo_data)));
        echo_data = echo_data + noise;
    end

    %% 5. 距离压缩（匹配滤波）
    % 对应理论笔记 §2.5：将 chirp 压缩为 sinc 尖峰，实现距离分辨
    %
    % 参考信号：发射 chirp 模板 exp(jπKr·t²)
    ref_signal = exp(1j*pi*Kr*fast_time.^2);

    % 匹配滤波器 h*(−t)：时间翻转 + 取复共轭
    %   时域卷积 x * h*(−t) 等价于 x 与 h 的互相关
    %   "对暗号"：回波与发射信号高度相关处产生大响应
    matched_filter = conj(fliplr(ref_signal));

    range_compressed = zeros(size(echo_data));
    for p = 1:num_pulses
        % 对每个脉冲（每列）的快时间数据做卷积
        % 'same'：输出截取为与输入等长，保持矩阵尺寸不变
        range_compressed(:, p) = conv(echo_data(:, p), matched_filter, 'same');
    end

    %% 6. 方位压缩（沿慢时间维 FFT）
    % 对应理论笔记 §2.6：将多普勒频率差异转化为空间（方位）分辨率
    %
    % fft(..., [], 2)：沿第2维（列/慢时间维）做 FFT
    %   每行（同一距离单元的 2001 个慢时间采样）→ 多普勒频域
    %   旋转角度不同 → 多普勒频率不同 → FFT 后在列方向分开
    %
    % fftshift(..., 2)：零频移至中心，使图像对称（负频在左，正频在右）
    isar_image = fftshift(fft(range_compressed, [], 2), 2);

end
