clc;
clear;
close all;
rng(42);

% ============================================================
% 0. 全局绘图设置
% ============================================================
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultLegendFontSize', 12);
set(groot, 'defaultLineLineWidth', 1);
set(groot, 'defaultLineMarkerSize', 6);
set(groot, 'defaultFigureColor', 'w');

% ============================================================
% 1. 生成单位球面网格并计算瑞利商
% ============================================================
intervals = 50;
ntheta = intervals;
nphi = 2 * intervals;
theta = linspace(0, pi, ntheta + 1);
phi = linspace(0, 2 * pi, nphi + 1);
r = 1;
[pp_, tt_] = meshgrid(phi, theta);
Z = r * cos(tt_);
X = r * sin(tt_) .* cos(pp_);
Y = r * sin(tt_) .* sin(pp_);
Points = [X(:), Y(:), Z(:)];

Q = [1, 0.5, 1; 
    0.5, 2, -0.2; 
    1, -0.2, 1];
Rayleigh_Q = sum((Points * Q) .* Points, 2);
Rayleigh_Q_ = reshape(Rayleigh_Q, size(X));

% ============================================================
% 2. 运行黎曼梯度下降算法并记录迭代轨迹
% ============================================================
x0 = [0.35; -0.88; 0.32];
x0 = x0 / norm(x0);
[x_opt, f_opt, info] = riemannian_gd_rayleigh_real(Q, x0, 1.0, 0.5, 1e-4, 1e-10, 500, 100, 1e-16);
x_hist = info.x_hist;
[theta_hist, phi_hist] = xyz_to_theta_phi(x_hist);

[V, D] = eig(Q);
lambda_all = diag(D);
[lambda_min, idx_min] = min(lambda_all);
v_min = V(:, idx_min);
direction_corr = abs(dot(x_opt, v_min));

fprintf('算法状态: %s\n', info.message);
fprintf('迭代次数: %d\n', info.iter);
fprintf('估计得到的最小瑞利商: %.12f\n', f_opt);
fprintf('真实最小特征值: %.12f\n', lambda_min);
fprintf('最终点与最小特征向量的方向相关性: %.12f\n', direction_corr);

% x_opt 和 v_min 是实特征向量，方向只差一个符号，因此需要符号对齐
if dot(x_opt, v_min) < 0
    x_aligned = -x_opt;
else
    x_aligned = x_opt;
end

eigvec_err = norm(x_aligned - v_min);
direction_corr = abs(dot(x_opt, v_min) / (norm(x_opt) * norm(v_min)));

fprintf('符号对齐后的特征向量误差: %.4e\n', eigvec_err);
fprintf('最终点与最小特征向量的方向相关性: %.12f\n', direction_corr);
fprintf('RGD 得到的最终点 x_opt:\n');
disp(x_opt);
fprintf('最小特征值对应的特征向量 v_min:\n');
disp(v_min);
fprintf('符号对齐后的 x_aligned:\n');
disp(x_aligned);

% ============================================================
% 3. 原始三维图 1：球面网格 + 彩色散点 + 截面 + RGD 轨迹
% ============================================================
fig = figure(1);
set(fig, 'Position', [100, 100, 900, 900]);
ax = axes(fig);
hold(ax, 'on');

mesh(ax, X, Y, Z, 'EdgeColor', [0.68, 0.68, 0.68], 'LineWidth', 0.25, 'FaceColor', 'none');
Xs = X(1:2:end, 1:2:end);
Ys = Y(1:2:end, 1:2:end);
Zs = Z(1:2:end, 1:2:end);
Cs = Rayleigh_Q_(1:2:end, 1:2:end);
scatter3(ax, Xs(:), Ys(:), Zs(:), 15, Cs(:), 'filled', 'HandleVisibility', 'off');

colormap(ax, hsv);

level_idx = 0.5;
[xx_, zz_] = meshgrid(linspace(-1.5, 1.5, 2), linspace(-1.5, 1.5, 2));
surf(ax, xx_, xx_ * 0 + level_idx, zz_, 'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
mesh(ax, xx_, xx_ * 0 + level_idx, zz_, 'EdgeColor', 'b', 'LineWidth', 0.2, 'FaceColor', 'none');

t_line = linspace(0, 2 * pi, 300);
r_line = sqrt(1 - level_idx^2);
plot3(ax, r_line * cos(t_line), level_idx * ones(size(t_line)), r_line * sin(t_line), 'b--', 'LineWidth', 2);

plot_trajectory_3d(ax, x_hist);

xlabel(ax, '$\it{x_1}$', 'Interpreter', 'latex');
ylabel(ax, '$\it{x_2}$', 'Interpreter', 'latex');
zlabel(ax, '$\it{x_3}$', 'Interpreter', 'latex');
set(ax, 'XTick', [], 'YTick', [], 'ZTick', []);
k_axis = 1.5;
plot3(ax, [-k_axis, k_axis], [0, 0], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [-k_axis, k_axis], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [0, 0], [-k_axis, k_axis], 'k', 'HandleVisibility', 'off');
axis(ax, 'off');
xlim(ax, [-max(abs(X(:))), max(abs(X(:)))]);
ylim(ax, [-max(abs(Y(:))), max(abs(Y(:)))]);
zlim(ax, [-max(abs(Z(:))), max(abs(Z(:)))]);
axis(ax, 'equal');
view(ax, -155, 35);
grid(ax, 'off');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 4. 原始三维图 2：球面颜色填充图 + RGD 轨迹
% ============================================================
fig = figure(2);
set(fig, 'Position', [100, 100, 1000, 1000]);
ax = axes(fig);
hold(ax, 'on');

surf(ax, X, Y, Z, Rayleigh_Q_, 'EdgeColor', 'none', 'FaceColor', 'interp', 'HandleVisibility', 'off');
colormap(ax, turbo);
plot_trajectory_3d(ax, x_hist);

xlabel(ax, '$\it{x_1}$', 'Interpreter', 'latex');
ylabel(ax, '$\it{x_2}$', 'Interpreter', 'latex');
zlabel(ax, '$\it{x_3}$', 'Interpreter', 'latex');
set(ax, 'XTick', [], 'YTick', [], 'ZTick', []);
k_axis = 1.5;
plot3(ax, [-k_axis, k_axis], [0, 0], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [-k_axis, k_axis], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [0, 0], [-k_axis, k_axis], 'k', 'HandleVisibility', 'off');
axis(ax, 'off');
xlim(ax, [-k_axis, k_axis]);
ylim(ax, [-k_axis, k_axis]);
zlim(ax, [-k_axis, k_axis]);
axis(ax, 'equal');
view(ax, -130, 30);
grid(ax, 'off');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 5. 原始三维图 3：透明球面网格图 + RGD 轨迹
% ============================================================
fig = figure(3);
set(fig, 'Position', [100, 100, 1000, 1000]);
ax = axes(fig);
hold(ax, 'on');

surf(ax, X, Y, Z, Rayleigh_Q_, 'FaceColor', 'interp', 'EdgeColor', 'interp', 'LineWidth', 0.25, 'FaceAlpha', 0, 'HandleVisibility', 'off');
colormap(turbo);
plot_trajectory_3d(ax, x_hist);

xlabel(ax, '$\it{x_1}$', 'Interpreter', 'latex');
ylabel(ax, '$\it{x_2}$', 'Interpreter', 'latex');
zlabel(ax, '$\it{x_3}$', 'Interpreter', 'latex');
set(ax, 'XTick', [], 'YTick', [], 'ZTick', []);
k_axis = 1.5;
plot3(ax, [-k_axis, k_axis], [0, 0], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [-k_axis, k_axis], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [0, 0], [-k_axis, k_axis], 'k', 'HandleVisibility', 'off');
axis(ax, 'off');
xlim(ax, [-k_axis, k_axis]);
ylim(ax, [-k_axis, k_axis]);
zlim(ax, [-k_axis, k_axis]);
axis(ax, 'equal');
view(ax, -130, 30);
grid(ax, 'off');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 6. 原始二维图 1：球面展开散点图 + RGD 轨迹
% ============================================================
fig = figure(4);
ax = axes(fig);
hold(ax, 'on');

scatter(ax, pp_(:), tt_(:), 8, Rayleigh_Q_(:), 'filled', 'HandleVisibility', 'off');
colormap(ax, turbo);
plot_trajectory_2d(ax, phi_hist, theta_hist);

ylim(ax, [min(tt_(:)), max(tt_(:))]);
xlim(ax, [min(pp_(:)), max(pp_(:))]);
xlabel(ax, '$\phi$', 'Interpreter', 'latex');
ylabel(ax, '$\theta$', 'Interpreter', 'latex', 'Rotation', 0);
set(ax, 'XTick', linspace(0, 2 * pi, 5));
set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
set(ax, 'YTick', linspace(0, pi, 3));
set(ax, 'YTickLabel', {'0', '\pi/2', '\pi'});
set(ax, 'YDir', 'reverse');
axis(ax, 'equal');
grid(ax, 'on');
box(ax, 'off');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 7. 原始二维图 2：伪彩色网格图 + RGD 轨迹
% ============================================================
fig = figure(5);
ax = axes(fig);
hold(ax, 'on');

surf(ax, pp_, tt_, zeros(size(Rayleigh_Q_)), Rayleigh_Q_, 'EdgeColor', 'none', 'HandleVisibility', 'off');
view(ax, 2);
shading(ax, 'flat');
colormap(ax, turbo);
colorbar(ax);
plot_trajectory_2d(ax, phi_hist, theta_hist);

xlim(ax, [min(pp_(:)), max(pp_(:))]);
ylim(ax, [min(tt_(:)), max(tt_(:))]);
xlabel(ax, '$\phi$', 'Interpreter', 'latex');
ylabel(ax, '$\theta$', 'Interpreter', 'latex', 'Rotation', 0);
set(ax, 'XTick', linspace(0, 2 * pi, 5));
set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
set(ax, 'YTick', linspace(0, pi, 3));
set(ax, 'YTickLabel', {'0', '\pi/2', '\pi'});
set(ax, 'YDir', 'reverse');
axis(ax, 'equal');
grid(ax, 'on');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 8. 原始二维图 3：填充等高线图 + 白色等高线 + RGD 轨迹
% ============================================================
fig = figure(6);
ax = axes(fig);
hold(ax, 'on');

levels = linspace(min(Rayleigh_Q_(:)), max(Rayleigh_Q_(:)), 18);
contourf(ax, pp_, tt_, Rayleigh_Q_, levels, 'LineColor', 'none', 'HandleVisibility', 'off');
contour(ax, pp_, tt_, Rayleigh_Q_, levels, 'LineColor', 'w', 'HandleVisibility', 'off');
colormap(ax, turbo);
colorbar(ax);
plot_trajectory_2d(ax, phi_hist, theta_hist);

ylim(ax, [min(tt_(:)), max(tt_(:))]);
xlim(ax, [min(pp_(:)), max(pp_(:))]);
xlabel(ax, '$\phi$', 'Interpreter', 'latex');
ylabel(ax, '$\theta$', 'Interpreter', 'latex', 'Rotation', 0);
set(ax, 'XTick', linspace(0, 2 * pi, 5));
set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
set(ax, 'YTick', linspace(0, pi, 3));
set(ax, 'YTickLabel', {'0', '\pi/2', '\pi'});
set(ax, 'YDir', 'reverse');
axis(ax, 'equal');
grid(ax, 'on');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 9. 原始二维图 4：二维等高线图 + RGD 轨迹
% ============================================================
fig = figure(7);
ax = axes(fig);
hold(ax, 'on');

contour(ax, pp_, tt_, Rayleigh_Q_, levels, 'HandleVisibility', 'off');
colormap(ax, turbo);
colorbar(ax);
plot_trajectory_2d(ax, phi_hist, theta_hist);

ylim(ax, [min(tt_(:)), max(tt_(:))]);
xlim(ax, [min(pp_(:)), max(pp_(:))]);
xlabel(ax, '$\phi$', 'Interpreter', 'latex');
ylabel(ax, '$\theta$', 'Interpreter', 'latex', 'Rotation', 0);
set(ax, 'XTick', linspace(0, 2 * pi, 5));
set(ax, 'XTickLabel', {'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
set(ax, 'YTick', linspace(0, pi, 3));
set(ax, 'YTickLabel', {'0', '\pi/2', '\pi'});
set(ax, 'YDir', 'reverse');
axis(ax, 'equal');
grid(ax, 'on');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 10. 从二维 theta-phi 平面中提取等高线
% ============================================================
C = contourc(phi, theta, Rayleigh_Q_, levels);

% ============================================================
% 11. 原始三维图 4：球面颜色图 + 白色等高线 + RGD 轨迹
% ============================================================
fig = figure(8);
set(fig, 'Position', [100, 100, 900, 900]);
ax = axes(fig);
hold(ax, 'on');

surf(ax, X, Y, Z, Rayleigh_Q_, 'EdgeColor', 'none', 'FaceColor', 'interp', 'HandleVisibility', 'off');
colormap(ax, turbo);
plot_contours_on_sphere(ax, C, 'white', [], 1);
plot_trajectory_3d(ax, x_hist);

xlabel(ax, '$\it{x_1}$', 'Interpreter', 'latex');
ylabel(ax, '$\it{x_2}$', 'Interpreter', 'latex');
zlabel(ax, '$\it{x_3}$', 'Interpreter', 'latex');
set(ax, 'XTick', [], 'YTick', [], 'ZTick', []);
k_axis = 1.5;
plot3(ax, [-k_axis, k_axis], [0, 0], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [-k_axis, k_axis], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [0, 0], [-k_axis, k_axis], 'k', 'HandleVisibility', 'off');
axis(ax, 'off');
xlim(ax, [-k_axis, k_axis]);
ylim(ax, [-k_axis, k_axis]);
zlim(ax, [-k_axis, k_axis]);
axis(ax, 'equal');
view(ax, -130, 30);
grid(ax, 'off');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 12. 原始三维图 5：球面网格 + 彩色等高线 + RGD 轨迹
% ============================================================
fig = figure(9);
set(fig, 'Position', [100, 100, 900, 900]);
ax = axes(fig);
hold(ax, 'on');

mesh(ax, X, Y, Z, 'EdgeColor', [0.5, 0.5, 0.5], 'LineWidth', 0.5, 'FaceColor', 'none', 'HandleVisibility', 'off');
plot_contours_on_sphere(ax, C, [], levels, 3);
plot_trajectory_3d(ax, x_hist);

xlabel(ax, '$\it{x_1}$', 'Interpreter', 'latex');
ylabel(ax, '$\it{x_2}$', 'Interpreter', 'latex');
zlabel(ax, '$\it{x_3}$', 'Interpreter', 'latex');
set(ax, 'XTick', [], 'YTick', [], 'ZTick', []);
k_axis = 1.5;
plot3(ax, [-k_axis, k_axis], [0, 0], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [-k_axis, k_axis], [0, 0], 'k', 'HandleVisibility', 'off');
plot3(ax, [0, 0], [0, 0], [-k_axis, k_axis], 'k', 'HandleVisibility', 'off');
axis(ax, 'off');
xlim(ax, [-k_axis, k_axis]);
ylim(ax, [-k_axis, k_axis]);
zlim(ax, [-k_axis, k_axis]);
axis(ax, 'equal');
view(ax, -130, 30);
grid(ax, 'off');
legend(ax, 'Location', 'northeast');
hold(ax, 'off');

% ============================================================
% 13. 可选图 1：瑞利商随迭代次数的变化
% ============================================================
fig = figure(10);
ax = axes(fig);
plot(ax, 1:length(info.f_hist), info.f_hist, '-o', 'LineWidth', 1.8, 'MarkerSize', 3);
grid(ax, 'on');
xlabel(ax, 'Iteration');
ylabel(ax, 'Rayleigh quotient');

% ============================================================
% 14. 可选图 2：黎曼梯度范数随迭代次数的变化
% ============================================================
fig = figure(11);
ax = axes(fig);
semilogy(ax, 1:length(info.grad_hist), info.grad_hist, '-o', 'LineWidth', 1.8, 'MarkerSize', 3);
grid(ax, 'on');
xlabel(ax, 'Iteration');
ylabel(ax, 'Riemannian gradient norm');

% ============================================================
% 15. 可选图 3：Armijo 步长随迭代次数的变化
% ============================================================
if ~isempty(info.eta_hist)
    fig = figure(12);
    ax = axes(fig);
    semilogy(ax, 1:length(info.eta_hist), info.eta_hist, '-o', 'LineWidth', 1.8, 'MarkerSize', 3);
    grid(ax, 'on');
    xlabel(ax, 'Iteration');
    ylabel(ax, 'Step size');
end

% ============================================================
% 16. 可选图 4：每次迭代的 Armijo 回溯次数
% ============================================================
if ~isempty(info.bt_hist)
    fig = figure(13);
    ax = axes(fig);
    stem(ax, 1:length(info.bt_hist), info.bt_hist, 'LineWidth', 1.2);
    grid(ax, 'on');
    xlabel(ax, 'Iteration');
    ylabel(ax, 'Backtracking number');
end

% ============================================================
% 本地函数 1：瑞利商最小化问题的黎曼梯度下降算法
% ============================================================
function [x, fval, info] = riemannian_gd_rayleigh_real(Q, x0, eta0, rho, c, epsilon, Kmax, max_backtrack, eta_min)
Q = 0.5 * (Q + Q.');
x = x0(:);
x = x / norm(x);
x_hist = x.';
f_hist = zeros(Kmax, 1);
grad_hist = zeros(Kmax, 1);
eta_hist = zeros(Kmax, 1);
bt_hist = zeros(Kmax, 1);
exitflag = 2;
message = '达到最大迭代次数后停止。';
iter = Kmax;
for k = 1:Kmax
    f = real(x.' * Q * x);
    g = Q * x - f * x;
    g_norm = norm(g);
    f_hist(k) = f;
    grad_hist(k) = g_norm;
    if g_norm <= epsilon
        exitflag = 1;
        message = '梯度范数达到停止阈值，算法收敛。';
        iter = k;
        break;
    end
    eta = eta0;
    success = false;
    bt_used = 0;
    for bt = 1:max_backtrack
        x_new = x - eta * g;
        x_new = x_new / norm(x_new);
        f_new = real(x_new.' * Q * x_new);
        if f_new <= f - c * eta * g_norm^2
            success = true;
            bt_used = bt;
            break;
        end
        eta = rho * eta;
        if eta <= eta_min
            bt_used = bt;
            break;
        end
    end
    if ~success
        exitflag = 0;
        message = 'Armijo 线搜索由于数值精度问题停止。';
        iter = k;
        break;
    end
    x = x_new;
    x_hist = [x_hist; x.'];
    eta_hist(k) = eta;
    bt_hist(k) = bt_used;
end
fval = real(x.' * Q * x);
info.x_hist = x_hist;
info.f_hist = f_hist(1:iter);
info.grad_hist = grad_hist(1:iter);
info.eta_hist = eta_hist(1:max(iter - 1, 0));
info.bt_hist = bt_hist(1:max(iter - 1, 0));
info.iter = iter;
info.exitflag = exitflag;
info.message = message;
end

% ============================================================
% 本地函数 2：三维坐标转换为 theta-phi 坐标
% ============================================================
function [theta_hist, phi_hist] = xyz_to_theta_phi(x_hist)
x1 = x_hist(:, 1);
x2 = x_hist(:, 2);
x3 = x_hist(:, 3);
x3 = min(max(x3, -1), 1);
theta_hist = acos(x3);
phi_hist = atan2(x2, x1);
phi_hist = mod(phi_hist, 2 * pi);
end

% ============================================================
% 本地函数 3：在三维球面图上绘制 RGD 轨迹
% ============================================================
function plot_trajectory_3d(ax, x_hist)
    plot3(ax, x_hist(:, 1), x_hist(:, 2), x_hist(:, 3), 'k-o', 'LineWidth', 2.4, 'MarkerSize', 3, 'DisplayName', 'RGD path');
    scatter3(ax, x_hist(1, 1), x_hist(1, 2), x_hist(1, 3), 80, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'DisplayName', 'Start');
    scatter3(ax, x_hist(end, 1), x_hist(end, 2), x_hist(end, 3), 90, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'End');
end

% ============================================================
% 本地函数 4：在二维展开图上绘制 RGD 轨迹
% ============================================================
function plot_trajectory_2d(ax, phi_hist, theta_hist)
    phi_plot = phi_hist(:);
    theta_plot = theta_hist(:);
    idx_jump = find(abs(diff(phi_plot)) > pi, 1, 'first');
    if ~isempty(idx_jump)
        phi_plot = [phi_plot(1:idx_jump); NaN; phi_plot(idx_jump + 1:end)];
        theta_plot = [theta_plot(1:idx_jump); NaN; theta_plot(idx_jump + 1:end)];
    end
    plot(ax, phi_plot, theta_plot, 'k-o', 'LineWidth', 2.4, 'MarkerSize', 3, 'DisplayName', 'RGD path');
    scatter(ax, phi_hist(1), theta_hist(1), 70, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'DisplayName', 'Start');
    scatter(ax, phi_hist(end), theta_hist(end), 80, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'End');
end

% ============================================================
% 本地函数 5：将二维等高线映射回三维单位球面
% ============================================================
function plot_contours_on_sphere(ax, C, fixed_color, levels, line_width)
    idx = 1;
    cmap = turbo(256);
    while idx < size(C, 2)
        level = C(1, idx);
        num_points = C(2, idx);
        phi_i = C(1, idx + 1:idx + num_points);
        theta_i = C(2, idx + 1:idx + num_points);
        Z_i = cos(theta_i);
        X_i = sin(theta_i) .* cos(phi_i);
        Y_i = sin(theta_i) .* sin(phi_i);
        if ~isempty(fixed_color)
            plot3(ax, X_i, Y_i, Z_i, 'Color', fixed_color, 'LineWidth', line_width, 'HandleVisibility', 'off');
        else
            t = (level - min(levels)) / (max(levels) - min(levels));
            color_idx = max(1, min(256, round(1 + t * 255)));
            plot3(ax, X_i, Y_i, Z_i, 'Color', cmap(color_idx, :), 'LineWidth', line_width, 'HandleVisibility', 'off');
        end
        idx = idx + num_points + 1;
    end
end