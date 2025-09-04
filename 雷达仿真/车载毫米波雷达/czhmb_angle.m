%% https://mp.weixin.qq.com/s?__biz=MzE5ODQ2NTg0NA==&mid=2247483658&idx=1&sn=9868541769a372df826510e76696d64a&chksm=97c80b95b674b82c70e46b85f614723ad97cf38f26cd72d3a581b2b15e2c2dbcdfff04a4c30f&mpshare=1&scene=1&srcid=0901B09mMxwbZDgpUkwZ96Ey&sharer_shareinfo=82505249d67a5b2bf596c8fd086b3d61&sharer_shareinfo_first=82505249d67a5b2bf596c8fd086b3d61&exportkey=n_ChQIAhIQAoITafoYTAx8zKLCyB%2BGqhKfAgIE97dBBAEAAAAAAMPKBGG%2Fd9oAAAAOpnltbLcz9gKNyK89dVj026nnoW3RP2GHYrbXwaoBPyyGTw7utyk9lHXzw%2B3xJg3F%2BUEDwK6nvcU71%2FaCsr75gBsC2uAk9sapxPIX2vdTJj%2FEPtBDify4kI9XXCzPJ7BO7tLDzOv4%2FYW1NZ%2Byn8KA139UM8yRn5%2BKXAf923SVggCBPZ6ASPYsyZO90qrsqkMao%2BBiT8%2Fi60s2Aaw48%2Bwy%2F8AFAVWy9GhJoZQyJ%2Bd17VEHVEAWXIz5E3wCcRveWcY2NSk%2BaC3xmQaG%2FFH7LJr0%2BQFTmixRwu6a%2BApesfbNo5KkiJswQ1xIy8shMUrdW9KMIlvrm8NqxalyRP349ZYOoWXlUcioPNmI&acctmode=0&pass_ticket=iou1Pj6x5eV%2B%2BcOz2cwpf5UqvazmfOjtPtT29Nw5XCZOUclqftdnWoBZemBJuYHm&wx_header=0&poc_token=HKZJtmijjQec3_8Li5j43j_tdSXXiV4dm0hcnPoV
%% 雷达参数设置
clear; clc; close all;

% 基本参数
c = 3e8;                % 光速 (m/s)
fc = 77e9;              % 载波频率 (Hz)
lambda = c/fc;          % 波长 (m)
d = 0.5*lambda;         % 阵元间距 (m)
N_ant = 16;             % 阵元数量
snr_db = 20;            % 信噪比 (dB)

% 目标参数
target_az = [0, 5];     % 两个目标的方位角 (度)
target_el = [0, 0];     % 俯仰角 (度)
N_targets = length(target_az);

% 角度扫描范围
az_range = linspace(-60, 60, 256); % 方位角扫描范围 -60°~60°
el_range = 0;                      % 俯仰角固定为0°

%% 生成阵列接收信号
% 导向矢量矩阵 (N_ant x N_targets)
A = zeros(N_ant, N_targets);
for k = 1:N_targets
    % 计算方位角对应的波程差 (弧度)
    psi = 2*pi*d*sin(deg2rad(target_az(k)))/lambda;
    % 构建导向矢量 (指数项表示波程差引起的相位延迟)
    A(:, k) = exp(-1j*(0:N_ant-1)'*psi);
end

% 生成目标信号 (假设单位幅度)
s = ones(N_targets, 1);

% 生成接收信号 (添加高斯白噪声)
noise_power = 10^(-snr_db/10); % 噪声功率
noise = sqrt(noise_power/2)*(randn(N_ant, 1) + 1j*randn(N_ant, 1));
x = A*s + noise; % 接收信号向量

%% FFT算法实现
% 计算FFT波束形成
N_fft = 1024; % FFT点数
fft_result = fft(x, N_fft); % 对接收信号做FFT

% 将FFT结果映射到角度域
az_fft = asind(linspace(-1, 1, N_fft)); % 角度映射
P_fft = abs(fft_result).^2; % 功率谱

% 归一化并转换为dB
P_fft_db = 10*log10(P_fft/max(P_fft));

%% DBF算法实现
% 初始化DBF功率谱
P_dbf = zeros(size(az_range));

% 遍历所有角度
for i = 1:length(az_range)
    % 计算当前角度的导向矢量
    psi = 2*pi*d*sin(deg2rad(az_range(i)))/lambda;
    a = exp(-1j*(0:N_ant-1)'*psi);

    % DBF波束形成
    P_dbf(i) = abs(a' * x)^2;
end

% 归一化并转换为dB
P_dbf_db = 10*log10(P_dbf/max(P_dbf));

%% DML算法实现
% 参数设置
D = 2;                  % 目标数 (固定为2)
N_candidate = 512;      % 候选角度总数
step_coarse = 8;        % 粗搜索步长
grd = 8;                % 精搜索范围
az_candidate = linspace(-60, 60, N_candidate); % 候选角度集

% 第一步：粗搜索
start_idx = randi([1, step_coarse]); % 随机起始点 (避免局部最优)
coarse_idx = start_idx:step_coarse:N_candidate;
N_coarse = length(coarse_idx);

% 初始化粗搜索结果
P_coarse = zeros(N_coarse, N_coarse);
max_val = -inf;
max_idx = [1, 1];

for i = 1:N_coarse
    for j = 1:N_coarse
        theta1 = az_candidate(coarse_idx(i));
        theta2 = az_candidate(coarse_idx(j));

        psi1 = 2*pi*d*sin(deg2rad(theta1))/lambda;
        psi2 = 2*pi*d*sin(deg2rad(theta2))/lambda;
        A_dml = [exp(-1j*(0:N_ant-1)'*psi1), ...
                 exp(-1j*(0:N_ant-1)'*psi2)];

        P_A = A_dml / (A_dml' * A_dml) * A_dml';

        R_hat = x * x';
        P_dml = real(trace(P_A * R_hat)); % 取实部

        % 存储结果并更新最大值max_idx_refine
        P_coarse(i, j) = P_dml;
        if P_dml > max_val
            max_val = P_dml;
            max_idx = [i, j];
        end
    end
end

% 提取DML估计角度
dml_theta1 = az_candidate(refine_idx_i(max_idx_refine(1)));
dml_theta2 = az_candidate(refine_idx_j(max_idx_refine(2)));

%% 结果可视化
% FFT和DBF结果比较
figure('Position', [100, 100, 1200, 500]);
subplot(1, 2, 1);
plot(az_fft, P_fft_db, 'LineWidth', 1.5);
hold on;
plot(az_range, P_dbf_db, 'LineWidth', 1.5);
xline(target_az(1), '--', 'Color', [0.5 0.5 0.5]);
xline(target_az(2), '--', 'Color', [0.5 0.5 0.5]);
xlim([-10, 15]);
xlabel('方位角 (度)');
ylabel('归一化功率 (dB)');
title('FFT vs DBF 角度分辨率比较');
legend('FFT', 'DBF', '真实角度');
grid on;

% DML精搜索结果可视化
subplot(1, 2, 2);
imagesc(az_candidate(refine_idx_j), az_candidate(refine_idx_i), P_refine);
hold on;
scatter(dml_theta2, dml_theta1, 100, 'rx', 'LineWidth', 2);
plot(target_az(2), target_az(1), 'wo', 'MarkerSize', 10, 'LineWidth', 2);
colorbar;
axis xy;
xlabel('目标2角度 (度)');
ylabel('目标1角度 (度)');
title(sprintf('DML算法估计结果: %.2f°, %.2f°', dml_theta1, dml_theta2));
set(gca, 'FontSize', 12);

% 添加分辨率分析文本
resolution_fft = sum(P_fft_db > max(P_fft_db)-3) * (az_fft(2)-az_fft(1));
resolution_dbf = sum(P_dbf_db > max(P_dbf_db)-3) * (az_range(2)-az_range(1));
fprintf('===== 分辨率分析结果 =====\n');
fprintf('目标真实角度: %.1f° 和 %.1f° (角度差: %.1f°)\n', target_az(1), target_az(2), abs(diff(target_az)));
fprintf('FFT算法分辨率: %.2f°\n', resolution_fft);
fprintf('DBF算法分辨率: %.2f°\n', resolution_dbf);
fprintf('DML估计角度: %.2f° 和 %.2f° (误差: %.2f°)\n', ...
        dml_theta1, dml_theta2, abs([dml_theta1, dml_theta2] - target_az));
fprintf('DML角度差估计: %.2f° (真实差: %.1f°)\n', ...
        abs(dml_theta1 - dml_theta2), abs(diff(target_az)));