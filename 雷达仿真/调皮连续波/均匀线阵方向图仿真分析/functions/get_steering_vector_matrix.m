function svm = get_steering_vector_matrix(ula, theta)
% GET_STEERING_VECTOR_MATRIX 获取导向向量矩阵
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算均匀线阵在多个角度的导向向量矩阵
% 版本: 1.0
% 输入参数:
%   ula - 均匀线阵结构体
%   theta - 角度数组 [rad]
% 输出参数:
%   svm - 导向向量矩阵 [N × N_angles]

    n = (0:ula.nr-1)';
    % 使用元素乘法，与Python保持一致
    svm = exp(2i * pi * ula.d * n .* sin(theta));
end