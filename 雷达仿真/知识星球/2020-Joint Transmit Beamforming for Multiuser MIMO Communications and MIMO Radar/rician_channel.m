function [ H ] = rician_channel( K_factor, M, K_users, theta_users, is_LoS )
% 空地 Rician 信道模型
%
% 输入:
%   K_factor    — Rician K 因子 (dB), 典型值 5~15 dB
%   M           — 发射天线数
%   K_users     — 用户数
%   theta_users — 用户方向角 (rad), 长度 K_users
%   is_LoS      — 是否含 LoS 分量 (逻辑, 默认 true)
%
% 输出:
%   H           — K_users × M 信道矩阵

if nargin < 5
    is_LoS = true;
end

K_linear = 10^(K_factor / 10);  % dB → 线性

% LoS 分量: 基于导向矢量的确定性信道
if is_LoS && exist('theta_users', 'var') && ~isempty(theta_users)
    H_LoS = zeros(K_users, M);
    for k = 1:K_users
        H_LoS(k, :) = ULA_steering_vector(M, theta_users(k))';
    end
else
    % 无 LoS 信息时退化为全1的秩1矩阵
    H_LoS = ones(K_users, M) / sqrt(M);
end

% NLoS 分量: 瑞利衰落
H_NLoS = (randn(K_users, M) + 1i * randn(K_users, M)) / sqrt(2);

% Rician 组合
H = sqrt(K_linear / (1 + K_linear)) * H_LoS ...
  + sqrt(1 / (1 + K_linear)) * H_NLoS;

end
