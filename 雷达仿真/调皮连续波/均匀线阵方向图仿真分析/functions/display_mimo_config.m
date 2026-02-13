function display_mimo_config(config)
%DISPLAY_MIMO_CONFIG 显示MIMO配置信息
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 在命令行窗口打印MIMO天线阵列配置信息
% 版本: 1.0
% 输入参数:
%   config - MIMO配置结构体

    fprintf('\n========== MIMO配置参数 ==========\n');
    fprintf('发射天线数量: %d\n', size(config.txl, 1));
    fprintf('接收天线数量: %d\n', size(config.rxl, 1));
    fprintf('虚拟阵元数量: %d\n', size(config.txl, 1) * size(config.rxl, 1));
    fprintf('波长: %.4f mm\n', config.lambda_m * 1000);
    fprintf('归一化间距: %.2f λ\n', config.d);
    fprintf('实际间距: %.4f mm\n', config.d_m * 1000);
    fprintf('====================================\n\n');
end
