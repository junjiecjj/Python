function [tau_est, fd_est, alpha_est] = Cal_single_2D_target_MUSIC(M, freq_axis, time_axis, c, lambda_c, prm, sparse_relative_positions,mask)
%Cal_single_2D_target_MUSIC 使用2D-MUSIC算法从信道矩阵中估计单个目标的参数
%
%   本函数实现了无空间平滑的2D-MUSIC算法
%   适用于单目标(K=1)或协方差矩阵秩为1的情况。
%
%   输入:
%       M:          [num_total_sc x num_sym] 复数信道响应矩阵。
%       freq_axis:  [num_total_sc x 1] 包含每个子载波频率的向量 (Hz)。
%       time_axis:  [1 x num_sym] 包含每个OFDM符号采样时刻的向量 (s)。
%       c:          光速 (m/s)。
%       lambda_c:   载波波长 (m)。
%       prm:        包含SIB1配置和OFDM信息的结构体。
%       mask:       [num_total_sc x num_sym] 用于幅度估计的有效点掩码。
%
%   输出:
%       tau_est:    估计出的双程延迟 (s)。
%       fd_est:     估计出的多普勒频移 (Hz)。
%       alpha_est:  估计出的复幅度。

%% --- 1. 数据预处理：提取用于MUSIC算法的密集矩阵 ---
SIB1_wavegenConfig = prm.SIB1_wavegenConfig;
PDSCH_fre_indices = (SIB1_wavegenConfig.PDSCH{1, 1}.PRBSet(1)*12+1):(SIB1_wavegenConfig.PDSCH{1, 1}.PRBSet(end)+1)*12;

% a. 合并所有需要使用的子载波索引
all_sc_indices = unique([PDSCH_fre_indices(:); sparse_relative_positions(:)]);
all_sc_indices = sort(all_sc_indices); % 保持顺序

% b. 提取形成密集矩形的数据矩阵 M_music
M_music = M(all_sc_indices, :);
[num_sc_music, num_sym_music] = size(M_music);

% c. 提取对应的物理坐标轴
freq_axis_music = freq_axis(all_sc_indices);
time_axis_music = time_axis(:); % 确保为列向量
T_sym = time_axis(2) - time_axis(1); % 符号周期

% d. 将截取后的矩阵向量化
y = M_music(:);

%% --- 2. 计算协方差矩阵 (单快照) ---
R_yy = y * y';

%% --- 3. 特征分解 ---
[E, D] = eig(R_yy);
[~, eig_indices] = sort(diag(D), 'descend');
E_sorted = E(:, eig_indices);

%% --- 4. 划分噪声子空间 ---
K = 1; % 假设为单目标
if size(E_sorted, 2) <= K
    error('无法划分噪声子空间，有效数据点数量过少。');
end
E_n = E_sorted(:, K+1:end);
En_H_En = E_n * E_n';

%% --- 5. 谱峰搜索 (在物理单位上直接搜索) ---
grid_res = 150;

% a. 定义时延和多普勒的物理搜索范围
range_max = 100; 
tau_max = 2 * range_max / c;
tau_search = linspace(0, tau_max, grid_res);

fd_max = 4000; % +/- 5kHz
fd_search = linspace(-fd_max, fd_max, grid_res);

music_spectrum = zeros(length(fd_search), length(tau_search));

% b. 迭代搜索
for i = 1:length(fd_search)
    for j = 1:length(tau_search)
        fd = fd_search(i);
        tau = tau_search(j);
        
        % --- 关键步骤：为密集矩阵构建导向矢量 a_vec ---
        % b_phi 对应时间/符号维度 (多普勒)
        % 使用物理时间轴构建
        b_phi = exp(1j * 2 * pi * fd * time_axis_music);
        
        % g_psi 对应频率/子载波维度 (时延)
        % 使用物理频率轴构建，以正确处理非连续子载波
        g_psi = exp(-1j * 2 * pi * tau * freq_axis_music).';
        
        % 由于 M_music 是密集矩形，现在可以安全地使用 kron
        % 顺序需匹配 M_music(:) 的向量化方式 (按列优先)
        a_vec = kron(b_phi, g_psi);
        
        % 计算 MUSIC 谱值
        denominator = a_vec' * En_H_En * a_vec;
        music_spectrum(i, j) = 1 / (abs(denominator) + eps); % 加微小量避免除零
    end
end

%% --- 6. 寻找最强的峰值 ---
[~, max_idx] = max(music_spectrum(:));
[fd_idx_est, tau_idx_est] = ind2sub(size(music_spectrum), max_idx);

fd_est = fd_search(fd_idx_est);
tau_est = tau_search(tau_idx_est);

range_est = tau_est * c / 2;
velocity_est = fd_est * lambda_c / 2;

%% --- 7. 幅度估计 ---
% 使用估计出的 tau 和 fd 在完整的原始数据上估计幅度
num_active_points = sum(mask, 'all');
steering_matrix = exp(-1j*2*pi*tau_est*freq_axis(:)) * exp(1j*2*pi*fd_est*time_axis(:)');
alpha_est = sum(M .* conj(steering_matrix) .* mask, 'all') / num_active_points;

end