function signal = get_signal(mva, X)
% GET_SIGNAL 提取阵列信号
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 提取并向量化MIMO阵列信号
% 版本: 1.0
% 输入参数:
%   mva - MIMO虚拟阵列结构体
%   X - 数据矩阵 [N_TX × N_RX × ...]
% 输出参数:
%   signal - 向量化信号

    if ndims(X) == 2
        % 按行展平（先转置再按列展平）
        signal = reshape(X.', [], 1);
    else
        sz = size(X);
        % 对前两维按行展平
        signal = reshape(permute(X, [2, 1, 3:ndims(X)]), [sz(1)*sz(2), sz(3:end)]);
    end
    signal = squeeze(signal);
end