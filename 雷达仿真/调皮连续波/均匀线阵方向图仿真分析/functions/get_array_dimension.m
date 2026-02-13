function dim = get_array_dimension(mva)
% GET_ARRAY_DIMENSION 获取阵列维度
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 获取MIMO虚拟阵列的维度
% 版本: 1.0
% 输入参数:
%   mva - MIMO虚拟阵列结构体
% 输出参数:
%   dim - 阵列维度(1或2)

    dim = mva.array_dimension;
end