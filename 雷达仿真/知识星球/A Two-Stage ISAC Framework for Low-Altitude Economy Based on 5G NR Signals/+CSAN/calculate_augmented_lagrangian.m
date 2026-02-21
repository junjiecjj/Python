function aug_lag = calculate_augmented_lagrangian(r_bar, S_tilde, z_bar, U, t, M, N, lambda, rho, Theta,  Upsilon)
% calculate_augmented_lagrangian
%
% 计算增广拉格朗日函数 (公式 33) 的当前值 (e_bar == 0 版本)

T_U = build_T(U, M, N); % 使用更新后的 U
M_matrix = [T_U, z_bar; z_bar', t]; % 使用更新后的 z, t

% --- 1. 计算原始目标函数值 (公式 26) ---
obj_original = calculate_objective(r_bar, S_tilde, z_bar, U, t, M, N, lambda);

% --- 2. 计算约束相关的项 ---

% 计算残差矩阵
residual_matrix = Theta - M_matrix;

% 计算对偶项 <Upsilon, Theta - M>
obj_dual = real(trace(Upsilon' * residual_matrix));

% 计算二次惩罚项 rho/2 * ||Theta - M||_F^2
obj_penalty = (rho / 2) * (norm(residual_matrix, 'fro')^2);

% --- 3. 求和 ---
% L_rho = (原始目标) + (对偶项) + (惩罚项)
% 注意: L_rho 中的 I_inf(Theta >= 0) 在 Theta 更新后总是为 0，所以我们不计算它
aug_lag = obj_original + obj_dual + obj_penalty;

end
