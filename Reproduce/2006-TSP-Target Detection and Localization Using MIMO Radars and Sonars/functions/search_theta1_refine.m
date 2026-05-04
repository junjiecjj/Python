function theta1_est = search_theta1_refine(eta, R_s, M, N, theta2_true, initial_range, final_tol, max_iters)
% 迭代细化网格搜索，第一次粗搜索，之后逐步缩小范围并加密网格
% 输入：
%   eta           - 等效观测向量 (M^2 x 1)
%   R_s           - 发射相干矩阵 (M x M)
%   M             - 阵元数
%   N             - 快拍数
%   theta2_true   - 目标2真实角度（度）
%   initial_range - 初始搜索半径（度），默认1.0
%   final_tol     - 最终搜索半径（度），小于此值停止，默认0.01
%   max_iters     - 最大迭代次数，默认5
% 输出：
%   theta1_est    - 估计的目标1角度（度）
    if nargin < 6, initial_range = 1.0; end
    if nargin < 7, final_tol = 0.01; end
    if nargin < 8, max_iters = 100; end

    range = initial_range;
    % 初始网格点数（每个维度），可适当粗糙
    grid_points = 21;   % 奇数，保证中心点
    theta1_center = 0;
    theta2_center = theta2_true;

    for iter = 1:max_iters
        theta1_grid = linspace(theta1_center - range, theta1_center + range, grid_points);
        theta2_grid = linspace(theta2_center - range, theta2_center + range, grid_points);
        best_val = -inf;
        best_th1 = NaN;
        best_th2 = NaN;

        for i = 1:length(theta1_grid)
            th1 = theta1_grid(i);
            for j = 1:length(theta2_grid)
                th2 = theta2_grid(j);
                D = construct_D([th1, th2], R_s, M, N);
                P_D = D * ((D'*D) \ D');
                L_val = real(eta' * P_D * eta);
                if L_val > best_val
                    best_val = L_val;
                    best_th1 = th1;
                    best_th2 = th2;
                end
            end
        end
        theta1_center = best_th1;
        theta2_center = best_th2;
        % 缩小范围并加密网格
        range = range / 2;
        grid_points = max(round(grid_points * 1.5), 11);  % 逐步加密，但保持奇数
        if range < final_tol
            break;
        end
    end
    theta1_est = theta1_center;
end