function [ null_dB ] = eval_null_depth( R, theta_jammer, theta_range, M )
% 评估指定方向上的零陷深度
%
% 输入:
%   R            — 协方差矩阵
%   theta_jammer — 干扰机方向 (rad)
%   theta_range  — 全角度范围 (用于归一化参考)
%   M            — 天线数
%
% 输出:
%   null_dB      — 零陷深度 (dB)，负值越大表示零陷越深

a_jammer = ULA_steering_vector(M, theta_jammer);
P_jammer = real(a_jammer' * R * a_jammer);

% 参考功率：方向图平均功率
a_all = ULA_steering_vector(M, theta_range);
P_avg = mean(real(diag(a_all' * R * a_all)));

null_dB = 10 * log10(P_jammer / P_avg);

end
