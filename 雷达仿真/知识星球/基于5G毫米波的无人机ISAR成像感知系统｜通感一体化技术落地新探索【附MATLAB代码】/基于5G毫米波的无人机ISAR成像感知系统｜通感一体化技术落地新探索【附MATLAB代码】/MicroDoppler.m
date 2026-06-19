%% 微多普勒特征提取
% 对应Python文件：27_MicroDoppler.py
% 用途：提取无人机旋翼引起的微多普勒时频特征
% 运行：MicroDoppler

function MicroDoppler()
    fc = 28e9; c = 3e8; PRF = 1000; T_obs = 2.0;

    fprintf('===== 微多普勒特征提取演示 =====\n\n');

    %% 三种不同无人机
    names = {'小型四旋翼', '大型六旋翼', '固定翼+单螺旋桨'};
    num_rotors = [4, 6, 1];
    rpms = [8000, 5000, 12000];
    blade_lens = [0.10, 0.20, 0.15];

    figure('Position', [100, 100, 1400, 800]);

    for col = 1:3
        fprintf('生成 %s 回波 (旋翼=%d, 转速=%dRPM)...\n', ...
            names{col}, num_rotors(col), rpms(col));

        echo = generate_rotor_echo(num_rotors(col), rpms(col), fc, ...
            PRF, T_obs, blade_lens(col), c);

        [S_db, f_axis, t_axis] = micro_doppler_stft_func(echo, PRF, 64);

        % 时频图
        subplot(2, 3, col);
        imagesc(t_axis, f_axis, S_db);
        axis xy; colorbar; colormap jet;
        caxis([max(S_db(:))-40, max(S_db(:))]);
        xlabel('时间 (s)'); ylabel('多普勒频率 (Hz)');
        title(names{col});

        % 平均谱
        subplot(2, 3, col+3);
        avg_spec = mean(S_db, 2);
        plot(f_axis, avg_spec, 'b-', 'LineWidth', 1);
        xlabel('多普勒频率 (Hz)'); ylabel('平均功率 (dB)');
        title([names{col}, ' - 平均谱']); grid on;
    end

    sgtitle('不同无人机类型的微多普勒特征对比', 'FontWeight', 'bold');
    saveas(gcf, 'figures/micro_doppler.png');
    fprintf('\n演示完成\n');
end

%% ==================== STFT ====================

function [S_db, f_axis, t_axis] = micro_doppler_stft_func(echo_signal, fs, window_len)
    overlap = floor(window_len * 0.75);
    nfft = 256;
    step = window_len - overlap;
    num_frames = floor((length(echo_signal) - overlap) / step);

    S = zeros(nfft, num_frames);
    win = hamming(window_len)';

    for i = 1:num_frames
        start_idx = (i-1) * step + 1;
        frame = echo_signal(start_idx:start_idx+window_len-1) .* win;
        S(:, i) = fftshift(fft(frame, nfft));
    end

    S_db = 20*log10(abs(S) + eps);
    f_axis = linspace(-fs/2, fs/2, nfft);
    t_axis = linspace(0, length(echo_signal)/fs, num_frames);
end

%% ==================== 旋翼回波生成 ====================

function echo = generate_rotor_echo(num_rotors, rotor_speed_rpm, fc, ...
                                     PRF, T_obs, blade_length, c)
    lambda = c / fc;
    omega_rotor = 2*pi*rotor_speed_rpm/60;
    num_pulses = round(T_obs * PRF);
    t = (0:num_pulses-1) / PRF;

    echo = zeros(1, num_pulses);

    % 机身（静止）
    echo = echo + 1.0 * exp(1j*rand()*2*pi);

    % 旋翼
    for r = 1:num_rotors
        phase_offset = 2*pi*(r-1)/num_rotors;
        echo = echo + 0.3 * exp(1j*(4*pi*blade_length/lambda) * ...
            sin(omega_rotor*t + phase_offset));
    end

    % 噪声 (SNR=10dB)
    snr_db = 10;
    noise_power = mean(abs(echo).^2) * 10^(-snr_db/10);
    noise = sqrt(noise_power/2) * (randn(1,num_pulses) + 1j*randn(1,num_pulses));
    echo = echo + noise;
end
