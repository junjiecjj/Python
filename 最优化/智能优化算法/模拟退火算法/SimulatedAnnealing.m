% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247483937&idx=1&sn=05c658f753e186104af15b91d4422d5a&chksm=c01cff6e76d3807478b8003ce237db50fbb909aa46d9f55dd385c8b77ba13c558e2c0284c211&mpshare=1&scene=1&srcid=0901pEkjpa6LWo9lcdz9marz&sharer_shareinfo=3c9cbe65c04aa521828e233ad71baeeb&sharer_shareinfo_first=3c9cbe65c04aa521828e233ad71baeeb&exportkey=n_ChQIAhIQWz8hUHcM5%2BC1vZJpC3wf3xKfAgIE97dBBAEAAAAAAHD0Iffl0VsAAAAOpnltbLcz9gKNyK89dVj0uQm4%2BaPCqq5WUBXlCAstHG1Fl8sGrXR0H7Sy1FR0qFN2MIK9ZJqxn9qtaYg8Zlqsk3Ynt0NgIeZpa%2BuFlnxML7rb7uOtJcnOkJIA0xxxkoIoyNQnAHV3qf6UVB0WQ%2BgGPdrEV03yPN9RLoyDMeWo340T8WdNfgigrQKsPeMBzpIlZR%2FHHdbXJdSU5eE2y69KApagbpH8lfFIwjtRLMMmiWZLmSB82oDQD%2FAAbuA9RI1nao1Nv2y47lZPYhwSQ78kWrczRcwSPP4DPyMx2h78jA%2FtZI6pzQdzSrSeddEZl0WBhs7JO%2BFWqP50TLHD8Fbeq3H7pKxgbaFW&acctmode=0&pass_ticket=pbvdnL9q8ORXrJ7up6iaKXGihuu6tpCgG9tZoUvAkQsBX%2FRJ0auIF71op4QUVxXz&wx_header=0#rd


clc
clear
close all

initial_solution = [0, 0];
initial_temp = 1000;
min_temp = 1e-3;
max_iter = 10000;

[best_solution, best_value, solution_history] = ...
    simulated_annealing(@objective_function, ...
    initial_solution, initial_temp, min_temp, max_iter);
fprintf('Best Solution: x = %f, y = %f\n', best_solution(1), best_solution(2));
fprintf('Best Value: %f\n', best_value);

[X, Y] = meshgrid(-6:0.1:6, -6:0.1:6);
Z = objective_function(X, Y);

figure;
subplot(1,2,1)
contourf(X, Y, Z, 50);
hold on;
plot(solution_history(:,1), solution_history(:,2), 'r.-', 'MarkerSize', 10);
plot(best_solution(1), best_solution(2), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
colorbar;
title('含搜索路径图');
xlabel('x');
ylabel('y');
legend('目标函数', '搜索路径', '最优解');
hold off;
subplot(1,2,2)
mesh(X,Y,Z)
title("FUNCTION FIGURE")
xlabel("x");
ylabel("y");
zlabel("value");


% 模拟退火算法的主函数
function [best_solution, best_value, solution_history] = ...
simulated_annealing(f, initial_solution,...
initial_temp, min_temp, max_iter)
    % 初始化
    current_solution = initial_solution;
    current_value = f(current_solution(1), current_solution(2));
    best_solution = current_solution;
    best_value = current_value;
    T = initial_temp;
    solution_history = current_solution;

    % 迭代过程
    for iter = 1:max_iter
        if T < min_temp
            break;
        end

        % 产生新解
        new_solution = neighbor_solution(current_solution);
        new_value = f(new_solution(1), new_solution(2));

        % 计算目标函数变化
        delta_E = new_value - current_value;

        % 决定是否接受新解
        if accept_solution(delta_E, T)
            current_solution = new_solution;
            current_value = new_value;

            % 更新最佳解
            if new_value < best_value
                best_solution = new_solution;
                best_value = new_value;
            end
        end

        % 记录解的历史
        solution_history = [solution_history; current_solution];

        % 降低温度
        T = cooling_schedule(T, iter);
    end
end

% 产生邻域解的函数
function new_solution = neighbor_solution(current_solution)
    % 在当前解的邻域内随机产生一个新解
    new_solution = current_solution + randn(1, 2);
end

% 判断是否接受新解的函数
function accept = accept_solution(delta_E, T)
    if delta_E < 0
        accept = true;
    else
        accept = rand() < exp(-delta_E / T);
    end
end

% 降温策略函数
function new_temp = cooling_schedule(T, iter)
    % 线性降温策略
    new_temp = T / (1 + 0.01 * iter);
end

% 目标函数
function value = objective_function(x, y)
    value = x^2 + y^2;
end


































