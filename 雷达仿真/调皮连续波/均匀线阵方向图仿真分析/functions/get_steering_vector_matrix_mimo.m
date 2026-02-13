function [svm, mask] = get_steering_vector_matrix_mimo(mva, theta)
% GET_STEERING_VECTOR_MATRIX_MIMO 获取MIMO虚拟阵列导向向量矩阵
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据阵列维度获取相应的导向向量矩阵
% 版本: 1.0
% 输入参数:
%   mva - MIMO虚拟阵列结构体
%   theta - 角度数组或元胞数组
% 输出参数:
%   svm  - 导向向量矩阵
%   mask - 有效角度掩码(二维阵列)

    if mva.array_dimension == 1
        svm = get_steering_vector_matrix(mva.la, theta);
        mask = [];
    else
        svm = get_steering_vector_matrix_2d(mva.pa, theta);
    end
    
    if ~mva.is_uniform
        svm = svm(mva.index_vec, :);
    end
    
    if mva.array_dimension == 2
        mask = get_valid_angle_mask(mva.pa, theta{2});
        svm = svm(:, mask(:));
    end
end