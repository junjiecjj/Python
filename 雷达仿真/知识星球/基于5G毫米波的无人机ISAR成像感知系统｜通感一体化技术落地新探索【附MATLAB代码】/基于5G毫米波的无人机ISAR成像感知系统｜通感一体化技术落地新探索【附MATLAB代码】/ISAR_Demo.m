%% ISAR成像演示 - 无人机雷达图像生成
% 用途：快速验证ISAR成像算法
% 运行方法：直接在MATLAB里运行即可

clear; clc; close all;
% clear  清除工作区所有变量，防止上次运行的残留数据干扰
% clc    清空命令窗口输出
% close all 关闭所有已打开的图形窗口

fprintf('====================================\n');
fprintf('  ISAR成像演示开始\n');
fprintf('====================================\n\n');

%% 1. 雷达参数设置
% 对应理论笔记 §2.2 雷达参数表
fc = 28e9;              % 载波频率 28GHz (5G毫米波 NR FR2频段)
c  = 3e8;               % 光速 (m/s)
lambda = c/fc;          % 波长 λ = c/fc ≈ 10.7mm，影响方位分辨率
B  = 400e6;             % 信号带宽 400MHz，决定距离分辨率 Δr = c/(2B) = 37.5cm
Tp = 1e-6;              % 脉冲宽度 1μs，chirp持续时间
fs = 2*B;               % 采样率 = 2倍带宽（奈奎斯特定理，防止混叠）
Kr = B/Tp;              % 调频率 Kr = B/Tp = 4×10^14 Hz/s，chirp频率变化速率

fprintf('雷达参数:\n');
fprintf('  载波频率: %.1f GHz\n', fc/1e9);
fprintf('  带宽: %.0f MHz\n', B/1e6);
fprintf('  距离分辨率: %.2f cm\n\n', c/(2*B)*100);
% 距离分辨率公式：Δr = c/(2B)，×100 换算为 cm

%% 2. 无人机目标模型（4个散射点）
% 对应理论笔记 §2.3：用4个点目标（散射中心）模拟十字形无人机
% 每行格式：[x, y, z]，单位：米；z 轴在2D成像中不使用
% 俯视图：
%       (0, 0.25)
%           |
%  (-0.3,0)---(0,0)---(0.3,0)
target_points = [
     0,    0,    0;      % 中心点（机身）
     0.3,  0,    0;      % 右臂（+x方向，0.3m）
    -0.3,  0,    0;      % 左臂（-x方向，0.3m）
     0,    0.25, 0       % 前臂（+y方向，0.25m）
];
num_points = size(target_points, 1);  % 散射点总数，此处 = 4

fprintf('目标设置:\n');
fprintf('  散射点数量: %d 个\n\n', num_points);

%% 3. 观测参数
% 对应理论笔记 §2.2 参数表后半部分
omega = 0.5;            % 旋转角速度 ω = 0.5 rad/s（无人机绕自身z轴自旋）
T_obs = 2;              % 观测时间 2秒，期间积累 PRF×T_obs = 2000 个脉冲
PRF   = 1000;           % 脉冲重复频率 1000Hz，即每秒发射1000个脉冲

% 慢时间轴：每个脉冲的发射时刻，步长 = 1/PRF = 1ms
% 结果为行向量 [0, 0.001, 0.002, ..., 2.000]，共 2001 个元素
slow_time  = 0:1/PRF:T_obs;
num_pulses = length(slow_time);  % = 2001

fprintf('观测参数:\n');
fprintf('  观测时间: %.1f 秒\n', T_obs);
fprintf('  脉冲数量: %d\n\n', num_pulses);

%% 4. 生成回波数据
% 对应理论笔记 §2.4 回波信号生成
fprintf('正在生成回波数据...\n');

% 快时间轴：单个脉冲内的采样时刻，步长 = 1/fs = 1.25ns
% 结果为行向量 [0, 1.25ns, 2.5ns, ..., 1μs]，共 801 个采样点
fast_time  = 0:1/fs:Tp;
num_samples = length(fast_time);  % = 801

% 预分配回波矩阵：行 = 快时间（距离维），列 = 慢时间（方位维）
% 矩阵大小 801×2001，复数矩阵
echo = zeros(num_samples, num_pulses);

R0 = 1000;  % 参考距离（无人机与基站初始距离）1000m

% ---- 外层循环：遍历每个脉冲（慢时间） ----
for p = 1:num_pulses
    % 第 p 个脉冲时刻，无人机已旋转的角度（弧度）
    % θ(t_m) = ω × t_m
    theta = omega * slow_time(p);

    % ---- 内层循环：遍历每个散射点 ----
    for k = 1:num_points
        % 旋转变换：将散射点原始坐标旋转 θ 角
        % 对应理论公式：x_k' = x_k·cosθ - y_k·sinθ
        %               y_k' = x_k·sinθ + y_k·cosθ
        x_rot = target_points(k,1)*cos(theta) - target_points(k,2)*sin(theta);
        y_rot = target_points(k,1)*sin(theta) + target_points(k,2)*cos(theta);

        % 散射点到基站的瞬时距离：R_k(t_m) = R0 + y_k'
        % y 方向为雷达视线方向（LOS），x 方向垂直于视线
        % 仅 y_rot 分量影响距离，x_rot 分量贡献方位（多普勒）信息
        R = R0 + y_rot;

        % 相对时延：τ_k = 2(R_k - R0)/c
        % 【关键】必须用相对时延而非绝对时延 2R/c：
        %   绝对时延 = 2×1000/(3×10^8) ≈ 6.67μs >> 快时间窗口上限 Tp=1μs
        %   若用绝对时延，信号完全落在采样窗口之外，回波全为零
        %   相对时延 ≈ ±2ns（散射点位移 ≤ 0.3m 引起），远小于 1μs，√
        tau = 2*(R - R0)/c;

        % 生成第 k 个散射点对第 p 个脉冲的回波（复信号）
        % 对应理论公式：s_k(t,t_m) = exp(jπKr(t-τ_k)²) × exp(-j4πf_c·R_k/c)
        %
        % 第一项 exp(jπKr(t-τ_k)²)：LFM chirp，τ_k 决定峰值位置（距离信息）
        %   (fast_time - tau) 是向量减标量，结果仍为向量（801元素）
        %   .^2 表示逐元素平方
        %
        % 第二项 exp(-j4πf_c·R/c)：载波相位项
        %   4π = 2×2π：因为信号往返（发射+接收各一次），相位翻倍
        %   R 用绝对距离（非相对），保留随 t_m 变化的多普勒相位信息
        %   这一项随慢时间 t_m 变化（因为 R 随旋转改变），FFT后产生多普勒分离
        sig = exp(1j*pi*Kr*(fast_time - tau).^2) .* ...
              exp(-1j*4*pi*fc*R/c);
        % sig 是行向量（1×801），需转置为列向量（801×1）才能累加到 echo 列
        echo(:, p) = echo(:, p) + sig';
    end
    % 每处理 400 个脉冲打印一次进度
    if mod(p, 400) == 0
        fprintf('  进度: %d/%d\n', p, num_pulses);
    end
end

% 加入高斯白噪声，SNR = 10dB（信噪比）
% 对应理论：s(t,t_m) = Σs_k(t,t_m) + n(t,t_m)
SNR_dB = 10;
try
    % 优先使用 Communications Toolbox 的 awgn 函数
    % 'measured' 表示以实测信号功率为基准计算噪声功率
    echo = awgn(echo, SNR_dB, 'measured');
catch
    % 无 Communications Toolbox 时手动添加
    signal_power = mean(abs(echo(:)).^2);               % 信号平均功率
    noise_power  = signal_power / (10^(SNR_dB/10));     % 由 SNR 反算噪声功率
    % 复高斯白噪声：实部虚部各贡献一半噪声功率，故各自方差 = noise_power/2
    % sqrt(noise_power/2) 是标准差，randn 生成标准正态分布随机数
    noise = sqrt(noise_power/2) * (randn(size(echo)) + 1j*randn(size(echo)));
    echo  = echo + noise;
end

fprintf('  回波数据生成完成！\n\n');

%% 5. 距离压缩（匹配滤波）
% 对应理论笔记 §2.5：将弥散的 chirp 压缩为尖锐峰，区分不同距离的散射点
fprintf('正在进行距离压缩...\n');

% 参考信号：发射 chirp 的理想模板，h(t) = exp(jπKr·t²)
ref_signal = exp(1j*pi*Kr*fast_time.^2);

% 匹配滤波器 = 参考信号的复共轭时间翻转：h*(−t)
% fliplr：左右翻转（时间反转）
% conj：取复共轭（*号）
% 物理含义：匹配滤波 = 接收信号与发射信号做"相关运算"
%   相关(x,y) = 卷积(x, y时间翻转后取共轭) = x * h*(−t)
matched_filter = conj(fliplr(ref_signal));

range_compressed = zeros(size(echo));
for p = 1:num_pulses
    % 对每列（每个脉冲的快时间数据）做卷积
    % 'same'：输出长度与输入相同（截取中间部分），保持矩阵尺寸不变
    % 卷积后，每个散射点在其对应距离单元产生 sinc 形状的尖峰
    range_compressed(:,p) = conv(echo(:,p), matched_filter, 'same');
end

fprintf('  距离压缩完成！\n\n');

%% 6. 方位压缩（FFT）
% 对应理论笔记 §2.6：沿慢时间维做 FFT，将多普勒频率差异转化为空间位置差异
fprintf('正在进行方位压缩...\n');

% fft(..., [], 2)：沿第2维（列方向，即慢时间/方位维）做 FFT
%   每一行（同一距离单元）的 2001 个慢时间采样 → 频域（多普勒域）
%   不同旋转位置的散射点具有不同多普勒频率，FFT 后在频域分开
%
% fftshift(..., 2)：将 FFT 结果的零频从左端移到中心
%   标准 FFT 输出：[0, 正频率..., 负频率...]
%   fftshift 后：  [负频率..., 0, 正频率...]（对称显示，便于观察）
isar_image = fftshift(fft(range_compressed, [], 2), 2);

fprintf('  方位压缩完成！\n\n');

%% 7. 显示结果
fprintf('正在生成图像...\n');

figure('Name', 'ISAR成像结果', 'Position', [100, 100, 1400, 500]);
set(gcf, 'Color', 'w');  % 图形背景设为白色

% 子图1：原始回波（复数取模，显示幅度）
subplot(1,3,1);
imagesc(abs(echo));      % abs 取复数的模（实部²+虚部²的平方根）
colormap(jet);
colorbar;
title('原始回波数据', 'FontSize', 14);
xlabel('慢时间（脉冲索引）', 'FontSize', 12);
ylabel('快时间（采样点）', 'FontSize', 12);

% 子图2：距离压缩后（dB 显示）
subplot(1,3,2);
% dB 转换：20log10(|x|/max|x| + eps)
%   先归一化（除以最大值），使峰值为 0dB
%   + eps 防止 log10(0) = -Inf（eps ≈ 2.2×10^-16 是 MATLAB 机器精度）
rc_norm = 20*log10(abs(range_compressed)/max(abs(range_compressed(:))) + eps);
imagesc(rc_norm);
colorbar;
title('距离压缩后（dB）', 'FontSize', 14);
xlabel('慢时间（脉冲索引）', 'FontSize', 12);
ylabel('距离单元', 'FontSize', 12);
caxis([-40 0]);          % 颜色轴范围：只显示 -40dB 到 0dB，抑制弱散射噪声

% 子图3：ISAR图像（最终成像结果）
subplot(1,3,3);
isar_norm = 20*log10(abs(isar_image)/max(abs(isar_image(:))) + eps);
imagesc(isar_norm);
colorbar;
title('ISAR成像结果（dB）', 'FontSize', 14);
xlabel('多普勒频率', 'FontSize', 12);  % 方位维：对应目标的旋转位置
ylabel('距离', 'FontSize', 12);
caxis([-40 0]);

% 保存图像（使用脚本所在目录，避免因当前目录不同导致路径错误）
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end  % 若在命令行直接运行则用当前目录
saveas(gcf, fullfile(script_dir, 'isar_result.png'));
fprintf('  图像已保存为 isar_result.png\n\n');

%% 8. 性能报告
fprintf('====================================\n');
fprintf('  成像性能总结\n');
fprintf('====================================\n');
fprintf('距离分辨率: %.2f cm\n', c/(2*B)*100);
% 距离分辨率：Δr = c/(2B) = 3e8/(2×400e6) = 0.375m = 37.5cm

fprintf('方位分辨率: %.2f cm\n', lambda/(2*omega*T_obs)*100);
% 方位分辨率：Δa = λ/(2ω·T_obs)
%   λ = c/fc ≈ 10.7mm，ω = 0.5 rad/s，T_obs = 2s
%   Δa = 0.0107/(2×0.5×2) = 0.00535m ≈ 0.535cm（理论值）

fprintf('图像大小: %d x %d\n', size(isar_image,1), size(isar_image,2));
% 行数 = 距离维采样点数（801），列数 = 方位维脉冲数（2001）
fprintf('====================================\n');
fprintf('演示完成！请查看生成的图像。\n');
fprintf('====================================\n');
