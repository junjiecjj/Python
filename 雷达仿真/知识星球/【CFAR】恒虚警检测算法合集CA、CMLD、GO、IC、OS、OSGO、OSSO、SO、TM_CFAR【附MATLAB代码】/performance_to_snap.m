%% 本文件用于模拟对数正态分布情况下不同经典算法的性能验证
clc;clear;close;

radar_points = 1024; % 雷达回波点数
fr = 1000; % 脉冲重复频率
lambda = 0.05;%波长
sigmav = 1.0;%速度方差
muc = 5; %尺度参数为10
sigmac = 0.6; %形状参数


target_num = 6; % 目标数目

% target_idx = [128, 256, 384, 420, 512];

target_idx = [randsample(100:200,1),randsample(200:300,1), randsample(300:400,1), randsample(400:500,1), randsample(500:600,1), ...
    randsample(600:700,1), randsample(700:800,1), randsample(800:900,1)];


% target_idx = randperm(length(clutter_data)); 
target_idx = target_idx(1:target_num); % 目标索引

target_test_window = 32; % 目标功率估计的测试单元
target_guard_window = 4; % 目标功率估计的保护窗

% CFAR基本参数设置
Pfa = [1e-6:2e-6:9e-6, 1e-5:2e-5:9e-5,1e-4:2e-4:9e-4,1e-3:2e-3:9e-3,1e-2:2e-2:9e-2,1e-1:2e-2:9e-2]; %虚警概率
NSlide = 64; % 参考单元数目
Pro_cell = 8; % 保护单元数目
rate = 3 / 4; % 有序选择序号

% 信噪比设置
snr_db = 10; % 信噪比设置
snr = 10 .^ (snr_db / 10); % 功率转化
monte_num = 1e3; % 蒙特卡洛仿真次数

% 检测矩阵初始化
Pd_CA = zeros(1, length(Pfa)); % CA-CFAR
Pd_SO = zeros(1, length(Pfa)); % SO-CFAR
Pd_GO = zeros(1, length(Pfa)); % GO-CFAR
Pd_OS = zeros(1, length(Pfa)); % OS-CFAR

tic
for pfa_id = 1 : length(Pfa)
    alpha_ca = ca_threhold(Pfa(pfa_id),NSlide); % CA门限系数
    alpha_so = so_threhold(Pfa(pfa_id),NSlide); % SO门限系数
    alpha_go = go_threhold(Pfa(pfa_id),NSlide); % GO门限系数
    alpha_os = os_threhold(Pfa(pfa_id),NSlide,rate); % OS门限系数
    ca_count = 0; % 初始化CA-CFAR统计值
    so_count = 0; % 初始化SO-CFAR统计值
    go_count = 0; % 初始化GO-CFAR统计值
    os_count = 0; % 初始化OS-CFAR统计值

    for monte_id = 1 : monte_num
        [clutter_data] = generate_lgclutter(radar_points, sigmav, muc, sigmac); % 每次随机模拟生成对数正态分布杂波
        
        for i = 1 : target_num % 生成特定snr下的多目标
            window_left = target_idx(i) - target_guard_window / 2 - target_test_window / 2;
            window_right = target_idx(i) + target_guard_window / 2 + target_test_window / 2;
            clutter_power = (mean(clutter_data(1, [window_left : window_left + target_test_window / 2 - 1])) + ...
                mean(clutter_data(1, [window_right - target_test_window / 2 + 1 : window_right]))) / 2;
            clutter_data(target_idx(i)) = clutter_power * snr;
        end
        
        
        % CA-CFAR算法
        ca_result = func_cfar_ca(clutter_data,alpha_ca,NSlide,Pro_cell);
        ca_result = cell2mat(ca_result(3));
        ca_length = length(ca_result);
        
        % SO-CFAR算法
        so_result = func_cfar_so(clutter_data,alpha_so,NSlide,Pro_cell);
        so_result = cell2mat(so_result(3));
        so_length = length(so_result);
        
        % GO-CFAR算法
        go_result = func_cfar_go(clutter_data,alpha_go,NSlide,Pro_cell);
        go_result = cell2mat(go_result(3));
        go_length = length(go_result);
        
        % OS-CFAR算法
        os_result = func_cfar_os(clutter_data,alpha_os,NSlide,Pro_cell,rate);
        os_result = cell2mat(os_result(3));
        os_length = length(os_result);
        
        % 统计不同算法的目标检测情况
        for tidx = 1 : target_num 
           for ca_idx = 1 : ca_length
               if ca_result(ca_idx) == target_idx(tidx)
                   ca_count = ca_count + 1;
                   break;
               end
           end
           
           for so_idx = 1 : so_length
               if so_result(so_idx) == target_idx(tidx)
                   so_count = so_count + 1;
                   break;
               end
           end
           
           for go_idx = 1 : go_length
               if go_result(go_idx) == target_idx(tidx)
                   go_count = go_count + 1;
                   break;
               end
           end
           
           for os_idx = 1 : os_length
               if os_result(os_idx) == target_idx(tidx)
                   os_count = os_count + 1;
                   break;
               end
           end
          
           
        end
    end
    
    % 统计每个snr下的目标检测概率
    Pd_CA(1, pfa_id) = ca_count / monte_num / target_num; 
    Pd_GO(1, pfa_id) = go_count / monte_num / target_num;
    Pd_SO(1, pfa_id) = so_count / monte_num / target_num;
    Pd_OS(1, pfa_id) = os_count / monte_num / target_num; 
end
toc

figure(1);
semilogx(Pfa, Pd_CA, 'Color','#FFD700','LineWidth',2.0,'Marker','d');hold on;
semilogx(Pfa, Pd_GO, 'Color','#483D8B','LineWidth',2.0,'Marker','+');hold on;
semilogx(Pfa, Pd_SO, 'Color','#FF1493','LineWidth',2.0,'Marker','*');hold on;
semilogx(Pfa, Pd_OS, 'Color','#8B008B','LineWidth',2.0,'Marker','s');hold on;
xlabel('\fontname{Times New Roman} False Alarm Rate');
ylabel('\fontname{Times New Roman} Pd');grid minor;
legend('CA-CFAR','GO-CFAR','SO-CFAR','OS-CFAR');
