%% Fig2_TIT.m
% 复现 TIT 论文 "CP-OFDM Achieves the Lowest Average Ranging Sidelobe ..." 的 Fig. 2
% 对比 SC, OFDM, CDMA 三种波形，有 CP（周期性 ACF），16-QAM，N=128
% 理论曲线：公式 (23)   仿真曲线：公式 (17)

clear; close all; clc; rng(42, 'twister');

% ---------- 绘图风格设置（与 Fig2.m 一致） ----------
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultTextFontSize',18);
set(groot,'defaultLineLineWidth',2);
set(groot,'defaultLineMarkerSize',6);
set(groot,'defaultFigureColor','white');
set(groot,'defaultLegendFontSize',18);

% ---------- 参数 ----------
N = 128;                 % 符号数 / 子载波数
M_const = 16;            % QAM 阶数
mu4 = 1.32;              % 16-QAM 的峰度（理论值）
num_MC = 500;            % 蒙特卡洛次数（可调，越大曲线越平滑）

% ---------- 生成 DFT 矩阵 ----------
FN = FFTmatrix(N);       % 归一化 DFT 矩阵

% ---------- 定义三种调制基 U ----------
% SC: 单位阵
U_SC = eye(N);
% OFDM: IDFT 矩阵（即 FN'）
U_OFDM = FN';
% CDMA: Hadamard 矩阵（N 必须为 2 的幂，128 满足）
U_CDMA = hadamard(N) / sqrt(N);   % 归一化保证酉性

waveforms = {'SC', 'OFDM', 'CDMA'};
U_cell = {U_SC, U_OFDM, U_CDMA};
colors = {[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250]}; % 蓝、橙、黄

% ---------- 生成 16-QAM 星座（单位功率） ----------
Constellation = qammod((0:M_const-1).', M_const, 'gray', 'UnitAveragePower', true);
% 确认峰度（可选）
% fprintf('实测峰度: %.4f\n', mean(abs(Constellation).^4));

% ---------- 预先计算 V = U' * FN' ----------
V_SC = U_SC' * FN';
V_OFDM = U_OFDM' * FN';
V_CDMA = U_CDMA' * FN';
V_cell = {V_SC, V_OFDM, V_CDMA};

% ---------- 理论计算（公式 23） ----------
E_theory = zeros(N, 3);   % 每列对应一种波形
for w = 1:3
    V = V_cell{w};
    for k = 0:N-1
        b_k = zeros(N, 1);
        for p = 1:N
            % 计算 b_k 的第 p 个元素：行向量 p 的 DFT
            row = V(p, :);          % 1×N
            abs_row_sq = abs(row).^2; % 1×N
            % DFT 求和：sum_{n=0}^{N-1} |v_{p,n}|^2 * exp(-j*2*pi*k*n/N)
            b_k(p) = sum(abs_row_sq .* exp(-1j * 2*pi * k * (0:N-1) / N));
        end
        norm_b2 = sum(abs(b_k).^2);
        % 公式 (23): E = N^2 * delta_{0,k} + N + (mu4-2)*norm_b2
        E = (k==0) * N^2 + N + (mu4 - 2) * norm_b2;
        E_theory(k+1, w) = E;
    end
end

% 转为 dB（避免主瓣过高，将主瓣也显示但会截断；此处全部显示）
E_theory_dB = 10*log10(E_theory + eps);

% ---------- 蒙特卡洛仿真（公式 17） ----------
E_sim = zeros(N, 3);
for w = 1:3
    U = U_cell{w};
    % 预分配累加器
    sum_sq = zeros(N, 1);
    for mc = 1:num_MC
        % 生成随机符号
        s = Constellation(randi(M_const, N, 1));
        x = U * s;                % 时域信号
        % 计算周期性 ACF 的所有延迟
        r = zeros(N, 1);
        for k = 0:N-1
            % 循环移位：J_k * x，即 x 循环右移 k 位（或左移，不影响 |r|^2）
            x_shift = circshift(x, k);
            r(k+1) = x' * x_shift;
        end
        sum_sq = sum_sq + abs(r).^2;
    end
    E_sim(:, w) = sum_sq / num_MC;
end
E_sim_dB = 10*log10(E_sim + eps);

% ---------- 绘图 ----------
x = 0:N-1;
figure('Position', [100, 100, 1200, 800]);
hold on;

for w = 1:3
    % 理论曲线（实线）
    plot(x, E_theory_dB(:, w), '-', 'Color', colors{w}, 'LineWidth', 2, ...
         'DisplayName', [waveforms{w} ' (Theory)']);
    % 仿真曲线（带圆圈标记）
    plot(x, E_sim_dB(:, w), 'o', 'Color', colors{w}, 'LineWidth', 1.5, ...
         'MarkerSize', 6, 'MarkerFaceColor', 'none', ...
         'DisplayName', [waveforms{w} ' (Sim)']);
end

xlabel('Delay Index k');
ylabel('Average Squared ACF (dB)');
xlim([0, N-1]);
% ylim 可根据需要调整（主瓣很高，可能自动缩放）
grid on;
legend('Location', 'northwest', 'EdgeColor', 'black');
box on;
hold off;

% 导出图片（可选）
% exportgraphics(gcf, 'Fig2_TIT.pdf', 'ContentType', 'vector');

%% 辅助函数：生成归一化 DFT 矩阵
function mat = FFTmatrix(L)
    mat = complex(zeros(L, L));
    for i = 0:L-1
        for j = 0:L-1
            mat(i+1, j+1) = exp(-1j * 2 * pi * i * j / L) / sqrt(L);
        end
    end
end