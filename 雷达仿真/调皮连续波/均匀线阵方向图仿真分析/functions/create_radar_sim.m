function radar_sim = create_radar_sim(cc, mc)
% CREATE_RADAR_SIM 创建雷达仿真器结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含Chirp和MIMO配置的雷达仿真器结构体
% 版本: 1.0
% 输入参数:
%   cc - Chirp配置结构体
%   mc - MIMO配置结构体
% 输出参数:
%   radar_sim - 雷达仿真器结构体

    radar_sim.cc = cc;
    radar_sim.mc = mc;
end