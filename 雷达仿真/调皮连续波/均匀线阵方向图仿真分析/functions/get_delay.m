function d = get_delay(radar_sim, trajectory)
% GET_DELAY 计算传播时延
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算每个TX-RX路径的传播时延
% 版本: 1.0
% 输入参数:
%   radar_sim - 雷达仿真器结构体
%   trajectory - 目标轨迹 [3 × N_timesteps]
% 输出参数:
%   d - 时延矩阵 [N_TX × N_RX × N_timesteps]

    num_tx = size(radar_sim.mc.txl, 1);
    num_rx = size(radar_sim.mc.rxl, 1);
    num_steps = size(trajectory, 2);
    d = zeros(num_tx, num_rx, num_steps);
    
    % TX0位置 [3 × 1]
    tx0 = radar_sim.mc.d_m * radar_sim.mc.txl(1, :)' ;
    
    % 计算TX0到目标的距离和方向
    outward = trajectory - tx0;  % [3 × N]
    r0 = vecnorm(outward, 2, 1);  % [1 × N]
    tof0 = 2 * r0 / radar_sim.cc.speed_of_light;  % [1 × N]
    
    sin_theta0 = outward(3, :) ./ r0;  % [1 × N]
    sin_phi0_cos_theta0 = outward(1, :) ./ r0;  % [1 × N]
    
    for k = 1:num_tx
        tx_k = radar_sim.mc.d_m * radar_sim.mc.txl(k, :)' ;  % [3 × 1]
        for l = 1:num_rx
            rx_l = radar_sim.mc.d_m * radar_sim.mc.rxl(l, :)' + tx_k;  % [3 × 1]
            % 计算时延
            tof_kl = (sin_theta0 * rx_l(3) + sin_phi0_cos_theta0 * rx_l(1)) ...
                / radar_sim.cc.speed_of_light;  % [1 × N]
            total_tof = tof0 + tof_kl;  % [1 × N]
            % 将行向量赋值到第三维
            d(k, l, 1:num_steps) = total_tof;
        end
    end
end