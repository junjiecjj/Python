function music = create_music(rank_estimator)
% CREATE_MUSIC 创建MUSIC算法结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含秩估计函数的MUSIC算法结构体
% 版本: 1.0
% 输入参数:
%   rank_estimator - 秩估计函数句柄
%                    输入: 特征值向量
%                    输出: 信号子空间秩
% 输出参数:
%   music - MUSIC算法结构体

    music.rank_estimator = rank_estimator;
end