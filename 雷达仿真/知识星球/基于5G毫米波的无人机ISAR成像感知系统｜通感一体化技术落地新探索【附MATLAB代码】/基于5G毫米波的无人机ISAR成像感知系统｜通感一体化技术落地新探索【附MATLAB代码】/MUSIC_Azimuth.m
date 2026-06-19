%% MUSIC超分辨率方位估计
% 对应Python文件：21_MUSIC_Azimuth.py
% 用途：对ISAR距离压缩后某一距离单元的慢时间信号进行超分辨率方位估计
% 运行：MUSIC_Azimuth

function MUSIC_Azimuth()
    %% 雷达参数
    fc = 28e9;
    c = 3e8;
    lambda = c / fc;
    omega = 0.5;
    PRF = 1000;
    num_pulses = 2000;

    %% 三个散射点的真实方位位置 (m)
    true_positions = [-0.15, 0.0, 0.18];
    num_sources = length(true_positions);
    amplitudes = [1.0, 0.8, 1.2];

    fprintf('===== MUSIC超分辨率方位估计演示 =====\n\n');
    fprintf('真实散射点位置: [%.2f, %.2f, %.2f] m\n', true_positions);
    fprintf('散射点个数: %d\n', num_sources);
    fprintf('Rayleigh极限: %.2f mm\n', ...
        lambda / (2*omega*(num_pulses/PRF)) * 1000);

    %% 生成慢时间信号
    t = (0:num_pulses-1) / PRF;
    signal = zeros(1, num_pulses);
    for k = 1:num_sources
        fd = 2 * omega * true_positions(k) / lambda;
        signal = signal + amplitudes(k) * exp(1j*2*pi*fd*t);
    end

    % 添加噪声 (SNR = 15 dB)
    snr_db = 15;
    noise_power = mean(abs(signal).^2) * 10^(-snr_db/10);
    noise = sqrt(noise_power/2) * (randn(1,num_pulses) + 1j*randn(1,num_pulses));
    signal_noisy = signal + noise;

    %% MUSIC估计
    fprintf('\n运行MUSIC算法...\n');
    [music_spec, music_angles] = music_azimuth_func(signal_noisy.', ...
        num_sources, omega, fc, c, PRF);

    %% FFT估计
    fprintf('运行FFT方位估计...\n');
    [fft_spec, fft_angles] = fft_azimuth_func(signal_noisy, omega, fc, c, PRF);

    %% 绘图对比
    figure('Position', [100, 100, 800, 600]);

    subplot(2,1,1);
    plot(fft_angles*100, fft_spec, 'b-', 'LineWidth', 1); hold on;
    for k = 1:num_sources
        xline(true_positions(k)*100, 'r--', 'LineWidth', 1);
    end
    xlim([-50 50]); ylim([-40 5]);
    xlabel('方位位置 (cm)'); ylabel('归一化幅度 (dB)');
    title('传统FFT方位估计');
    legend('FFT谱', '真实位置'); grid on;

    subplot(2,1,2);
    plot(music_angles*100, music_spec, 'b-', 'LineWidth', 1); hold on;
    for k = 1:num_sources
        xline(true_positions(k)*100, 'r--', 'LineWidth', 1);
    end
    xlim([-50 50]); ylim([-40 5]);
    xlabel('方位位置 (cm)'); ylabel('伪谱幅度 (dB)');
    title('MUSIC超分辨率方位估计');
    legend('MUSIC伪谱', '真实位置'); grid on;

    sgtitle('MUSIC vs FFT 方位分辨率对比', 'FontWeight', 'bold');

    saveas(gcf, 'figures/music_vs_fft.png');
    fprintf('\n演示完成，结果已保存到 figures/music_vs_fft.png\n');
end

%% ==================== 核心算法函数 ====================

function [pseudo_spectrum, angles] = music_azimuth_func(signal, num_sources, omega, fc, c, PRF)
% MUSIC超分辨率方位估计
% signal:      num_pulses x 1 慢时间信号（单距离单元）
% num_sources: 散射点个数（需先验已知）
% omega, fc, c, PRF: 旋转角速度、载波频率、光速、PRF
% 返回: pseudo_spectrum 伪谱(dB), angles 扫描方位位置(m)

    lambda = c / fc;
    N = length(signal);
    M = floor(N / 3);  % 子阵长度

    % 1. 构建协方差矩阵（前向-后向平均提高估计精度）
    Rxx = zeros(M);
    for i = 1:(N - M + 1)
        x = signal(i:i+M-1);
        Rxx = Rxx + (x * x');
    end
    Rxx = Rxx / (N - M + 1);

    % 2. 特征分解：信号子空间 + 噪声子空间
    [V, D] = eig(Rxx);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    En = V(:, num_sources+1:end);  % 噪声子空间

    % 3. 扫描方位角，计算伪谱
    angles = linspace(-0.5, 0.5, 1000);
    pseudo_spectrum = zeros(size(angles));

    for i = 1:length(angles)
        fd = 2 * omega * angles(i) / lambda;
        a = exp(1j*2*pi*fd*(0:M-1)'/PRF);  % 导向矢量
        pseudo_spectrum(i) = 1 / abs(a' * (En * En') * a);
    end

    pseudo_spectrum = 10*log10(pseudo_spectrum / max(pseudo_spectrum));
end

function [spectrum_db, pos_axis] = fft_azimuth_func(signal, omega, fc, c, PRF)
% 传统FFT方位估计（对比基线）
    lambda = c / fc;
    nfft = 1024;
    spectrum = fftshift(fft(signal, nfft));
    spectrum_db = 20*log10(abs(spectrum) / max(abs(spectrum)) + 1e-20);

    freq_axis = linspace(-PRF/2, PRF/2, nfft);
    pos_axis = freq_axis * lambda / (2 * omega);
end
