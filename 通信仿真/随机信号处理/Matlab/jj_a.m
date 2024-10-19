


% 1 | 数据滤波：探讨卡尔曼滤波、SG滤波与组合滤波
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485974&idx=1&sn=19fd8a052e51ae5b1d58c90f0d003cd9&chksm=c15b9b38f62c122ef5740bcf06941e7faca8bac79ef054cd257c59f212dcf278849cac6298d5&cur_album_id=3587607448191893505&scene=190#rd


clc
clear
close all

function main
Q_vals = [0.1, 1, 10];
R_vals = [0.1, 1, 10];
dt = 1;
t = 0:dt:100;
n = length(t);
F = [1 dt; 0 1];
H = [1 0];
B = [0; 0];
u = 0;

num_experiments = length(Q_vals) * length(R_vals);
errs = cell(num_experiments, 1);
ests = cell(num_experiments, 1);
leg = cell(num_experiments, 1);
idx = 1;

% 生成真实轨迹
x_true = [0; 1];
x_true_hist = zeros(2, n);
z_hist = zeros(1, n);

for k = 1:n
    x_true = F * x_true + B * u + sqrt(1) * randn(2,1);
    z = H * x_true + sqrt(1) * randn;
    x_true_hist(:, k) = x_true;
    z_hist(k) = z;
end

for i = 1:length(Q_vals)
    for j = 1:length(R_vals)
        Q = Q_vals(i) * eye(2);
        R = R_vals(j);

        x_est = [0; 0];
        P_est = eye(2);
        x_est_hist = zeros(2, n);

        for k = 1:n
            [x_pred, P_pred] = predict(F, B, u, x_est, P_est, Q);
            [x_est, P_est] = update(x_pred, P_pred, H, z_hist(k), R);
            x_est_hist(:, k) = x_est;
        end

        err = sqrt(sum((x_true_hist - x_est_hist).^2, 1));
        errs{idx} = err;
        ests{idx} = x_est_hist;
        leg{idx} = sprintf('Q=%.1f, R=%.1f', Q_vals(i), R_vals(j));
        idx = idx + 1;
    end
end

plot_errors(t, errs, leg);
plot_trajectories(t, x_true_hist, z_hist, ests, leg);
end

function [x_pred, P_pred] = predict(F, B, u, x_est, P_est, Q)
x_pred = F * x_est + B * u;
P_pred = F * P_est * F' + Q;
end

function [x_est, P_est] = update(x_pred, P_pred, H, z, R)
K = P_pred * H' / (H * P_pred * H' + R);
x_est = x_pred + K * (z - H * x_pred);
P_est = (eye(size(K,1)) - K * H) * P_pred;
end

function plot_errors(t, errs, leg)
figure;
hold on;

for i = 1:length(errs)
    plot(t, errs{i}, 'LineWidth', 2);
end

legend(leg, 'Location', 'best');
title('误差随时间变化');
xlabel('时间');
ylabel('误差 (欧几里得范数)');
grid on;
hold off;
end

function plot_trajectories(t, x_true, z_hist, ests, leg)
leg_1 = ["真实位置"; "测量值"; leg];
figure;
%% 位置轨迹
subplot(2,1,1);
plot(t, x_true(1,:), 'g', 'LineWidth', 2); hold on;
plot(t, z_hist, 'rx', 'LineWidth', 1.5);

for i = 1:length(ests)
    plot(t, ests{i}(1,:), 'LineWidth', 2);
end
legend(leg_1, 'Location', 'best');

xlabel('时间');
ylabel('位置');
hold off;

%% 速度轨迹
leg_2 = ["真实速度"; leg];
subplot(2,1,2);
plot(t, x_true(2,:), 'g', 'LineWidth', 2); hold on;
for i = 1:length(ests)
    plot(t, ests{i}(2,:), 'LineWidth', 2);
end
legend(leg_2, 'Location', 'best');
title('速度估计');
xlabel('时间');
ylabel('速度');
hold off;
end

