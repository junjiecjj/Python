function signal = get_signal_padded(mva, X)
% GET_SIGNAL_PADDED 提取并填充阵列信号
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 提取MIMO阵列信号并填充到虚拟阵列掩码中
% 版本: 1.0
% 输入参数:
%   mva - MIMO虚拟阵列结构体
%   X - 数据矩阵 [N_TX × N_RX × ...]
% 输出参数:
%   signal - 填充后的信号矩阵

    if ndims(X) == 2
        signal = zeros(size(mva.va_mask), class(X));
        % 按行展平（先转置再按列展平）
        vec_X = reshape(X.', 1, []);
    else
        sz = size(X);
        signal = zeros([size(mva.va_mask), sz(3:end)], class(X));
        % 对前两维按行展平
        vec_X = reshape(permute(X, [2, 1, 3:ndims(X)]), [sz(1)*sz(2), sz(3:end)]);
    end
    
    % 使用线性索引进行赋值
    lin_idx = sub2ind(size(mva.va_mask), mva.index_mat{1}, mva.index_mat{2});
    signal(lin_idx) = vec_X;
    signal = squeeze(signal);
end