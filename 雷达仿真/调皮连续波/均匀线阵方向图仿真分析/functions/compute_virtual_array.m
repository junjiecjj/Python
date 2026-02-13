function va = compute_virtual_array(config)
%COMPUTE_VIRTUAL_ARRAY 计算虚拟阵列位置
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据TX和RX天线位置计算MIMO虚拟阵列位置
% 版本: 1.0
% 输入参数:
%   config - MIMO配置结构体，包含:
%            .txl - 发射天线位置矩阵 [N_TX × 3]
%            .rxl - 接收天线位置矩阵 [N_RX × 3]
% 输出参数:
%   va - 虚拟阵列位置矩阵 [N_TX × N_RX × 3]，单位为半波长

    num_tx = size(config.txl, 1);
    num_rx = size(config.rxl, 1);
    va = zeros(num_tx, num_rx, 3);
    
    for k = 1:num_tx
        for l = 1:num_rx
            va(k, l, :) = config.txl(k, :) + config.rxl(l, :);
        end
    end
end
