clear; clc; close all;
rng(42);

%% 通用参数
N = 4;          % 发射天线 / 目标维数
M = 6;          % 接收天线
L = N;          % 雷达快拍数（取 L=N 简化）
PT = 10;        % 总发射功率
sigma_c2 = 1;   % 高斯白噪声方差
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

%% 3. MIMO 容量（高斯白噪声）
fprintf('========== 3. MIMO 容量（高斯白噪声） ==========\n');
H = randn(M, N) + 1i*randn(M, N);
Sigma_Z = sigma_c2 * eye(M);
[V_H, Lambda_H] = eig(H'*H);
Lambda_H = real(diag(Lambda_H));
[Lambda_H, idx_H] = sort(Lambda_H, 'descend');
V_H = V_H(:, idx_H);

[P_white, water_level] = water_filling(sigma_c2, Lambda_H, PT);
fprintf('理论功率分配: %s\n', mat2str(P_white',4));

obj_white = @(s) -sum(log(1 + s(:).*Lambda_H / sigma_c2));
Aeq = ones(1,N); beq = PT;
lb = zeros(N,1);
s0 = PT/N * ones(N,1);
[s_opt, fval] = fmincon(obj_white, s0, [], [], Aeq, beq, lb, [], [], options);
P_white_num = s_opt';

fprintf('fmincon功率分配: %s\n', mat2str(P_white_num,4));
fprintf('绝对误差: %.2e\n', norm(P_white - P_white_num));

Sigma_X_theory = V_H * diag(P_white) * V_H';
Sigma_X_fmincon = V_H * diag(P_white_num) * V_H';
C_white = log(det(H * Sigma_X_theory * H' + Sigma_Z)) - M*log(sigma_c2);
C_white_num = log(det(H * Sigma_X_fmincon * H' + Sigma_Z)) - M*log(sigma_c2);
fprintf('容量理论: %.8f\n', C_white);
fprintf('容量fmincon: %.8f\n\n', C_white_num);

%% 4. MIMO 容量（色噪声）
fprintf('========== 4. MIMO 容量（色噪声） ==========\n');
Z = randn(M, L) + 1i*randn(M, L);
Sigma_Z = (Z * Z') / L;
[Uz, Lz] = eig(Sigma_Z);
Lz = real(diag(Lz));
[Lz, idx_z] = sort(Lz, 'descend');
Uz = Uz(:, idx_z);
Hw = Uz' * H;
[Vw, Lw] = eig(Hw'*Hw);
Lw = real(diag(Lw));
[Lw, idx_w] = sort(Lw, 'descend');
Vw = Vw(:, idx_w);

Lz_sel = Lz(end-N+1:end);
[Lw_asc, idx_asc] = sort(Lw, 'ascend');
Lz_sel_asc = Lz_sel(idx_asc);
[P_asc, water_level] = water_filling(Lz_sel_asc, Lw_asc, PT);
P_color = zeros(N,1);
P_color(idx_asc) = P_asc;
fprintf('理论功率分配: %s\n', mat2str(P_color',4));

obj_color = @(s) -sum(log(s(:).*Lw + Lz_sel));
Aeq = ones(1,N); beq = PT;
lb = zeros(N,1);
[s_opt, fval] = fmincon(obj_color, s0, [], [], Aeq, beq, lb, [], [], options);
P_color_num = s_opt';

fprintf('fmincon功率分配: %s\n', mat2str(P_color_num,4));
fprintf('绝对误差: %.2e\n', norm(P_color - P_color_num));

Sigma_X_theory = Vw * diag(P_color) * Vw';
Sigma_X_fmincon = Vw * diag(P_color_num) * Vw';
C_color = log(det(H * Sigma_X_theory * H' + Sigma_Z)) - log(det(Sigma_Z));
C_color_num = log(det(H * Sigma_X_fmincon * H' + Sigma_Z)) - log(det(Sigma_Z));
fprintf('容量理论: %.8f\n', C_color);
fprintf('容量fmincon: %.8f\n\n', C_color_num);
