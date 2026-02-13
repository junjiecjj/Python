function spectrum = get_music_spectrum(music, Rxx, Atheta, varargin)
% GET_MUSIC_SPECTRUM 计算MUSIC伪谱
% 作者: Radar Engineer
% 时间: 2026-02-13
% 功能: 使用MUSIC算法计算高分辨率角度伪谱
% 版本: 1.1
% 输入参数:
%   music  - MUSIC算法结构体
%   Rxx    - 协方差矩阵 [N × N]
%   Atheta - 导向向量矩阵 [N × N_angles]
%   varargin - 可选参数
%     'AngleDimensions' - 角度维度 [n_theta, n_phi]，用于二维阵列
% 输出参数:
%   spectrum - MUSIC伪谱
%             - 一维阵列: [N_angles × 1]
%             - 二维阵列: [n_theta × n_phi]

    N_angles = size(Atheta, 2);
    spectrum = zeros(N_angles, 1);
    
    % 特征值分解
    [V, D] = eig(Rxx);
    w = real(diag(D));
    
    % 特征值排序(降序)
    [w, idx] = sort(w, 'descend');
    V = V(:, idx);
    
    % 估计信号子空间秩
    rank = music.rank_estimator(w);
    
    % 提取噪声子空间
    En = V(:, rank+1:end);
    
    % 计算MUSIC伪谱
    for s = 1:N_angles
        a = Atheta(:, s);
        EnVs = En' * conj(a);
        spectrum(s) = (a' * conj(a)) / (EnVs' * EnVs);
    end
    
    spectrum = abs(spectrum);
    
    % 处理二维阵列的情况
    % 检查是否提供了角度维度
    angle_dims = [];
    for i = 1:2:length(varargin)
        if strcmp(varargin{i}, 'AngleDimensions')
            angle_dims = varargin{i+1};
            break;
        end
    end
    
    % 如果提供了角度维度，将谱reshape为二维
    if ~isempty(angle_dims) && length(angle_dims) == 2
        % 检查维度是否匹配
        if prod(angle_dims) == N_angles
            spectrum = reshape(spectrum, angle_dims);
        else
            warning('角度维度与谱长度不匹配，保持一维输出');
        end
    end
end