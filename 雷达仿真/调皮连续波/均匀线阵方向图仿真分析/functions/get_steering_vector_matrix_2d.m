function svm = get_steering_vector_matrix_2d(ura, theta_phi)
% GET_STEERING_VECTOR_MATRIX_2D 获取2D导向向量矩阵
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算均匀面阵在多个角度对的导向向量矩阵
% 版本: 1.0
% 输入参数:
%   ura - 均匀面阵结构体
%   theta_phi - 角度元胞数组 {theta_array, phi_array}
% 输出参数:
%   svm - 导向向量矩阵 [N_elements × N_angle_pairs]

    theta_arr = theta_phi{1};
    phi_arr = theta_phi{2};
    
    num_angles = length(theta_arr) * length(phi_arr);
    svm = zeros(prod(ura.nr), num_angles);
    
    k = 1;
    for i = 1:length(theta_arr)
        for j = 1:length(phi_arr)
            svm(:, k) = get_steering_vector_2d(ura, [theta_arr(i), phi_arr(j)]);
            k = k + 1;
        end
    end
end