%% ClutterGen.m 杂波建模函数
function [x_all, f_s, f_d, azimuth_c] = ClutterGen(H_c, R_t, v_c, azimuth_target, azimuth_array, ...
    N_bin, CPI_c, N_R, d, lambda, PRF, B, k_sam, sigma0, P_t, Ls)
    % 载机高度H_c（m），目标距离R_c（m），载机速度v_c（m/s），
    % H_c 载机高度(m)
    % R_t 目标距离(m)
    % v_c 载机速度(m/s)
    % azimuth_target 目标方位（°）
    % azimuth_array 阵列与载机飞行方向夹角（°）
    % N_bin 杂波块个数 
    % CPI_c CPI内脉冲数
    % N_R 阵元数
    % d 阵元间隔（m）
    % lambda 波长(m)
    % PRF 脉冲重复频率(Hz)
    % B 带宽(Hz)
    % k_sam 样本个数（杂波距离环个数）
    % sigma0 杂波后向散射系数
    % P_t 发射功率 (W)
    % Ls 接收机损耗
    % 函数返回杂波信号x_all （N_R×CPI，k_sam+1）

    % 常数设置
    c = 3e8;            % 光速 (m/s)
    H = H_c;      % 载机高度（m）
    v = v_c;            % 载机速度（m/s）
    L = CPI_c;          % CPI内脉冲数
    delta_R = c / (2 * B); % 距离环间隔

    % % 添加噪声功率计算
    % k_B = 1.38e-23; % 玻尔兹曼常数
    % T0 = 290;       % 噪声温度
    % P_noise = k_B * T0 * B; % 噪声功率
    % CNR_dB = 40;
    % CNR_linear = 10^(CNR_dB/10);        % 1e4
    
    %--------------计算待测距离环和参考距离环的俯仰角----------------
    R = R_t;                      % 目标距离（m）
    R_all = R + delta_R * (-k_sam/2 : k_sam/2); % 所有距离环距载机的距离
    phi_all = asin(H ./ R_all);          % 所有距离环俯仰角

    % 杂波块数目和方向角设置
    azimuth_c = linspace(-pi/2, pi/2, N_bin); % 杂波块方位角/正侧视[-90,90]/正前视[0,180]
    theta_rel = azimuth_c - deg2rad(azimuth_array); % 考虑阵列方向
    d_theta = pi / (N_bin - 1);     % 方位角间隔

    %--------------各距离环杂波块的空时频率------------
    f_s = (d/lambda) * cos(phi_all)' * sin([linspace(-pi/2, 0, (N_bin-1)/2), 0, linspace(0, pi/2, (N_bin-1)/2)]); % 各距离环每个杂波块的空间频率 
    f_d = (2*v/lambda) * cos(phi_all)' * sin([linspace(-pi/2, 0, (N_bin-1)/2), 0, linspace(0, pi/2, (N_bin-1)/2)]) / PRF; % 各距离环每个杂波块的多普勒频率

    Amplitude_all = zeros(k_sam+1,N_bin); % 各距离环各杂波块的复幅度
    x_all = zeros(N_R*L,k_sam+1);  % 所有距离环的杂波回波数据
    for ring_num = 1:length(R_all)
        R_ring = R_all(ring_num);
        phi = phi_all(ring_num);
        % 地面投影距离
        % R_ground = sqrt(R_ring^2 - H^2);
        R_ground = R_ring * cos(phi);

        %---------------计算各杂波块CNR和幅度---------------------
        area_patch = delta_R * R_ground * d_theta;
        RCS_patch = sigma0 * area_patch;
        
        % 雷达方程或者固定杂噪比计算幅度（天线方向图增益为1）
        Pr = (P_t * N_R^2 * lambda^2 * RCS_patch) / ((4*pi)^3 * R_ring^4 * Ls); % CNR×P_noise为杂波功率
        % Pr = CNR_linear * P_noise; % CNR×P_noise为杂波功率

        Amplitude_all(ring_num,:) = sqrt(Pr); % 根据雷达方程或预设杂噪比计算该距离环每个杂波块的复幅度
        
        % 空时导向矢量
        a_s = exp(1j*2*pi*(0:N_R-1)' * f_s(ring_num,:));
        a_t = exp(1j*2*pi*(0:L-1)' * f_d(ring_num,:));
        
        %-----------------------回波数据-----------------------
        for clutterpatch_num = 1:N_bin
            x_all(:,ring_num) = x_all(:,ring_num) + ...
            Amplitude_all(ring_num,clutterpatch_num) * ...
            kron(a_t(:,clutterpatch_num),a_s(:,clutterpatch_num));
            %该距离环各杂波块回波数据叠加
        end
    end
end
