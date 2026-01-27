%% ============= 归一化（每条曲线用自己的最大值）并绘图 ===================

% Ideal：用自身最大值归一化
norm_ideal       = max(P_ideal);
P_ideal_n        = P_ideal / (norm_ideal + eps);

% Radar-Only（Shared）：用雷达-only 方向图自身的最大值归一化
norm_radar_only  = max(P_radar_only);
P_radar_only_n   = P_radar_only / (norm_radar_only + eps);

% RadCom（Shared）：用 RadCom 方向图自身的最大值归一化
norm_radcom      = max(P_radcom);
P_radcom_n       = P_radcom / (norm_radcom + eps);

% 线性刻度
figure;
plot(theta_deg, P_ideal_n, 'k--','LineWidth',1.5); hold on;
plot(theta_deg, P_radar_only_n,'b-','LineWidth',1.5);
plot(theta_deg, P_radcom_n,'r-','LineWidth',1.5);
grid on; xlim([-90 90]);
xlabel('Angle (Degree)');
ylabel('Normalized Beampattern');
legend('Ideal','Radar-Only (Shared)','RadCom (Shared)','Location','Best');
title('Fig.3(b) Shared Deployment: Multi-beam Beampatterns (self-normalized)');
