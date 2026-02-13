function config = create_mimo_config(txl, rxl, lambda_m, d)
%CREATE_MIMO_CONFIG 创建MIMO天线阵列配置结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含发射和接收天线位置信息的MIMO配置结构体
% 版本: 1.0
% 输入参数:
%   txl      - 发射天线位置矩阵 [N_TX × 3]，单位为半波长
%   rxl      - 接收天线位置矩阵 [N_RX × 3]，单位为半波长
%   lambda_m - 波长 [m]
%   d        - 归一化单元间距(相对波长)，默认0.5
% 输出参数:
%   config - MIMO配置结构体，包含:
%            .txl      - 发射天线位置矩阵
%            .rxl      - 接收天线位置矩阵
%            .lambda_m - 波长 [m]
%            .d        - 归一化间距
%            .d_m      - 实际间距 [m]
%            .va       - 虚拟阵列位置 [N_TX × N_RX × 3]

    if nargin < 4
        d = 0.5;
    end
    
    config.txl = txl;
    config.rxl = rxl;
    config.lambda_m = lambda_m;
    config.d = d;
    config.d_m = d * lambda_m;
    
    config.va = compute_virtual_array(config);
end
