%% Riemannian Gradient Descent on Sphere for Rayleigh Quotient Minimization
%  Uses a general positive definite matrix (not diagonal).
%  Visualizes convergence path on a colored sphere.
%  Adjusts view angle to show the descent path clearly.

clear; clc; close all;

%% 1. Generate a general positive definite symmetric matrix
rng(42);   % for reproducibility
N = 3;
% Random matrix
M = randn(N);
% Make symmetric positive definite
A = M' * M + 1 * eye(N);
fprintf('Matrix A:\n'); disp(A);

% Rayleigh quotient function (full definition)
f = @(x) (x' * A * x) / (x' * x);

%% 2. Prepare sphere mesh and color according to f(x)
[Xs, Ys, Zs] = sphere(100);
x_vec = [Xs(:), Ys(:), Zs(:)]';
f_vals = zeros(1, size(x_vec,2));
for i = 1:size(x_vec,2)
    x = x_vec(:,i);
    f_vals(i) = (x' * A * x) / (x' * x);
end
F_grid = reshape(f_vals, size(Xs));

figure;
surf(Xs, Ys, Zs, F_grid, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
colormap(jet);
colorbar;
caxis([min(f_vals), max(f_vals)]);
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Rayleigh quotient on the sphere: f(x) = (x^T A x)/(x^T x)');
grid on;
hold on;

[V, D] = eig(A);
[~, idx_min] = min(diag(D));
v_min = V(:, idx_min);
%% 3. Riemannian gradient descent (minimization)
%% 迭代部分（带 Armijo 线搜索）
max_iter = 100;
mu_init = 1;       % 初始步长
beta = 0.5;        % 步长衰减因子
sigma = 0.01;      % 充分下降参数
tol = 1e-6;

x0 = randn(N,1); x0 = x0 / norm(x0);
x = x0;
X_hist = x;

for iter = 1:max_iter
    f_val = f(x);
    grad_euc = 2 * A * x - 2 * f_val * x;
    grad_riem = grad_euc - (x' * grad_euc) * x;
    
    if norm(grad_riem) < tol
        break;
    end
    
    % 搜索方向（负梯度）
    d = -grad_riem;
    gamma = real(x' * d);   % 由于 x'*d = 0，gamma=0，此处需要重新计算方向上的方向导数
    % 正确的方向导数：沿 d 的方向导数为 grad_riem' * d = -norm(grad_riem)^2
    dir_deriv = -norm(grad_riem)^2;   % 负值
    
    % Armijo 线搜索
    mu = mu_init;
    while true
        x_trial = x + mu * d;
        x_trial = x_trial / norm(x_trial);
        f_trial = f(x_trial);
        if f_trial <= f_val + sigma * mu * dir_deriv
            break;
        end
        mu = mu * beta;
    end
    
    x = x + mu * d;
    x = x / norm(x);
    X_hist = [X_hist, x];
end

fprintf('Final f(x) = %f\n', f(x));
fprintf('Final point (approx eigenvector of smallest eigenvalue):\n');
disp(x);

%% 4. Plot trajectory and adjust view
x_vals = X_hist(1,:);
y_vals = X_hist(2,:);
z_vals = X_hist(3,:);

% Plot path
plot3(x_vals, y_vals, z_vals, 'k-', 'LineWidth', 2);
% Start point (green)
scatter3(x_vals(1), y_vals(1), z_vals(1), 100, 'g', 'filled', 'MarkerEdgeColor', 'k');
% End point (red)
scatter3(x_vals(end), y_vals(end), z_vals(end), 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
% True minimizer (black diamond, both antipodal points)
scatter3(v_min(1), v_min(2), v_min(3), 80, 'k', 'filled', 'Marker', 'diamond');
scatter3(-v_min(1), -v_min(2), -v_min(3), 80, 'k', 'filled', 'Marker', 'diamond');

legend('Sphere', 'Iteration path', 'Start', 'End', 'True minimizer', 'Location', 'best');

% Adjust view angle to see the descent path clearly
% Azimuth = 45°, Elevation = 20° (usually works)
view(45, 20);
% Enable interactive rotation if needed:
% rotate3d on;

hold off;