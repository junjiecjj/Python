% https://mp.weixin.qq.com/s?__biz=MzE5ODQ2NTg0NA==&mid=2247483950&idx=1&sn=8c3ef3f43e14e1c2ca7d5e755dab0752&chksm=97152431f5de62ad4481c912e5b0e1ea7b3aaa66393240cf10996b5fbefd8abc5ad7b07d9a81&mpshare=1&scene=1&srcid=1013NiErdgX4tUejQzIN63eX&sharer_shareinfo=76a8c2d378e6cac6d27a63f78aef2d4b&sharer_shareinfo_first=76a8c2d378e6cac6d27a63f78aef2d4b&exportkey=n_ChQIAhIQ1tSUT82pZvRkwL%2B7yVp1lhKfAgIE97dBBAEAAAAAACzSB5uw0dcAAAAOpnltbLcz9gKNyK89dVj0DGN2lWoTghECYjddoV7fGkrXZXDEDSh28hb1C2zjaJilhRKM5v4O377FkrHFQLwxq834ksjtR755rFNAcWzq%2BfXQysX2Kh9ioOtr4IlYMg9JL7xIaJ364SqfnJBmkzXJbwY7LPl1B0u1LLVOu51hcY2Y%2F90oc%2BvmzY1fNIFI3Q7dvxQatCXLSCdEwjTAFDN4k%2BZWNqOxkIZwIkwe3L6nc7Mdhmyq8H4%2Bfvw35uZVpifIS%2B1FKDYJ5Wmic9y2c2zqZn5QlCCnK%2Bk3O5ifTeYznJggS%2Bm3aNuH5yYeZ70kqzroHQtFuYmyBwKNbgZ3cxN37Pt4FonnZYdV&acctmode=0&pass_ticket=QG1YqDDkdHYeaquYFYcruJKXk6KxJjz%2BvOC7fzcoI5HtG90jCHQFa6V2pba8dy5S&wx_header=0#rd

clear; close all; clc;
%  雷达参数设置

%% 参数设置
c = 3e8;            % 光速 (m/s)
fc = 77e9;          % 载波频率 (Hz)
lambda = c/fc;      % 波长 (m)

% 雷达参数
PRF = 2000;         % 脉冲重复频率 (Hz)
T = 1/PRF;          % 脉冲重复间隔 (s)
N = 512;            % 每个脉冲的采样点数
CPI = 0.5;          % 相干处理间隔 (s)
M = round(CPI/T);   % 脉冲数

% 目标参数
target_types = {'行人', '自行车', '汽车'};
num_targets = length(target_types);

target_types = {'行人', '自行车', '汽车'};
num_targets = length(target_types);

%% 生成微多普勒特征
% 初始化微多普勒特征矩阵
microdoppler_features = cell(1, num_targets);
time_axis = (0:M-1)*T;
frequency_axis = (-M/2:M/2-1)*(PRF/M);

% 对于每种目标类型生成微多普勒特征
for target_idx = 1:num_targets
    target_type = target_types{target_idx};

    % 根据目标类型设置参数
    switch target_type
        case '行人'
            % 行人参数
            v_main = 1.5;           % 主体速度 (m/s)
            v_arm = 0.8;            % 手臂摆动速度 (m/s)
            v_leg = 1.2;            % 腿部摆动速度 (m/s)
            f_arm = 1.2;            % 手臂摆动频率 (Hz)
            f_leg = 1.8;            % 腿部摆动频率 (Hz)
            arm_amp = 0.3;          % 手臂摆动幅度 (m)
            leg_amp = 0.4;          % 腿部摆动幅度 (m)

        case '自行车'
            % 自行车参数
            v_main = 5;             % 主体速度 (m/s)
            v_pedal = 2;            % 踏板速度 (m/s)
            wheel_rpm = 200;        % 车轮转速 (RPM)
            f_pedal = 1.5;          % 踏板频率 (Hz)
            pedal_amp = 0.15;       % 踏板幅度 (m)
            wheel_amp = 0.3;        % 车轮幅度 (m)

        case '汽车'
            % 汽车参数
            v_main = 15;            % 主体速度 (m/s)
            wheel_rpm = 300;        % 车轮转速 (RPM)
            wheel_amp = 0.35;       % 车轮幅度 (m)
            engine_vib_freq = 30;   % 发动机振动频率 (Hz)
            engine_vib_amp = 0.02;  % 发动机振动幅度 (m)
    end

    % 生成微多普勒信号
    microdoppler_signal = zeros(M, N);

    for m = 1:M
        t = (m-1)*T;

        % 根据目标类型计算微多普勒频率
        switch target_type
            case '行人'
                % 行人微多普勒模型
                arm_velocity = v_arm * sin(2*pi*f_arm*t);
                leg_velocity = v_leg * sin(2*pi*f_leg*t + pi/4);
                microdoppler_freq = 2/lambda * (v_main + arm_velocity + leg_velocity);

            case '自行车'
                % 自行车微多普勒模型
                pedal_velocity = v_pedal * sin(2*pi*f_pedal*t);
                wheel_velocity = (wheel_rpm * 2*pi/60) * wheel_amp * sin(2*pi*(wheel_rpm/60)*t);
                microdoppler_freq = 2/lambda * (v_main + pedal_velocity + wheel_velocity);

            case '汽车'
                % 汽车微多普勒模型
                wheel_velocity = (wheel_rpm * 2*pi/60) * wheel_amp * sin(2*pi*(wheel_rpm/60)*t);
                engine_vibration = engine_vib_amp * sin(2*pi*engine_vib_freq*t);
                microdoppler_freq = 2/lambda * (v_main + wheel_velocity + engine_vibration);
        end

        % 生成信号
        phase = 2*pi*microdoppler_freq*t;
        microdoppler_signal(m, :) = exp(1j*phase);
    end

    % 添加噪声
    SNR = 20; % 信噪比 (dB)
    microdoppler_signal = awgn(microdoppler_signal, SNR, 'measured');

    % 存储微多普勒特征
    microdoppler_features{target_idx} = microdoppler_signal;
end

%3.3 时频分析 - 生成微多普勒谱
figure('Position', [100, 100, 1200, 800]);
sgtitle('不同目标的微多普勒特征', 'FontSize', 16, 'FontWeight', 'bold');

for target_idx = 1:num_targets
    % 计算短时傅里叶变换(STFT)
    signal = microdoppler_features{target_idx};
    [s, f, t] = spectrogram(mean(signal, 2), 64, 60, 1024, PRF);

    % 绘制微多普勒谱
    subplot(2, 2, target_idx);
    imagesc(t, f/1e3, 20*log10(abs(s)));
    axis xy;
    xlabel('时间 (s)');
    ylabel('多普勒频率 (kHz)');
    title([target_types{target_idx} '的微多普勒谱']);
    colorbar;
    clim([-50, 0]);
    colormap('jet');
end

%3.4 特征提取与分类
% 提取微多普勒特征用于分类
features = [];
labels = [];

for target_idx = 1:num_targets
    signal = microdoppler_features{target_idx};

    % 计算多普勒频谱
    doppler_spectrum = fftshift(fft(signal, M, 1), 1);

    % 提取特征
    % 1. 频谱中心
    spectrum_center = sum(abs(doppler_spectrum).*frequency_axis', 1) ./ sum(abs(doppler_spectrum), 1);

    % 2. 频谱带宽
    spectrum_bandwidth = sqrt(sum(abs(doppler_spectrum).*(frequency_axis'.^2), 1) ./ sum(abs(doppler_spectrum), 1) - spectrum_center.^2);

    % 3. 频谱熵
    norm_spectrum = abs(doppler_spectrum) ./ sum(abs(doppler_spectrum), 1);
    spectrum_entropy = -sum(norm_spectrum .* log(norm_spectrum + eps), 1);

    % 4. 频谱峰度
    spectrum_kurtosis = kurtosis(abs(doppler_spectrum), 1);

    % 组合特征
    target_features = [mean(spectrum_center); mean(spectrum_bandwidth); 
                      mean(spectrum_entropy); mean(spectrum_kurtosis)];

    features = [features, target_features];
    labels = [labels, target_idx];
end

% 显示提取的特征
feature_names = {'频谱中心', '频谱带宽', '频谱熵', '频谱峰度'};
figure('Position', [100, 100, 1000, 400]);
for i = 1:4
    subplot(2, 2, i);
    bar(features(i, :));
    set(gca, 'XTickLabel', target_types);
    title(feature_names{i});
    ylabel('特征值');
end
sgtitle('不同目标的微多普勒特征比较', 'FontSize', 16, 'FontWeight', 'bold');





