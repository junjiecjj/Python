function sv = get_steering_vector_mimo(mva, theta)
% GET_STEERING_VECTOR_MIMO 获取MIMO虚拟阵列导向向量
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据阵列维度获取相应的导向向量
% 版本: 1.0
% 输入参数:
%   mva - MIMO虚拟阵列结构体
%   theta - 角度 [rad] 或 [theta, phi]
% 输出参数:
%   sv - 导向向量

    if mva.array_dimension == 1
        sv = get_steering_vector(mva.la, theta);
    else
        sv = get_steering_vector_2d(mva.pa, theta);
    end
    
    if ~mva.is_uniform
        sv = sv(mva.index_vec);
    end
end