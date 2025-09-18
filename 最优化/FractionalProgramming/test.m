clc; clear; cvx_clear;
cvx_expert true;

rng(1) % 随机种子
epsilon = 1e-5; % 收敛阈值
Max_iter = 50; % 最大迭代步数

R = 2; % 基站到基站的距离 0.8
PL = @(d) 128.1 + 37.6*log10(d); % 路损模型，d--km
U = 7; % 用户个数，每个蜂窝中有一个用户
C = 7; % 基站个数
P = 50; % 最大发射功率43 dBm
sigma2 = -105; % 噪声功率 -100 dBm
shadowing_std = 8; % 阴影衰落的标准差-8 dB

B = 10e6; % 10Mhz

H_gain = channel_generate(U,R,PL,shadowing_std); % 随机产生用户位置，并产生信道增益
H_gain = db2pow(H_gain - sigma2); % 为了优化方便，将噪声归一化

% 直接式求解方法
p_temp = db2pow(P*ones(C,1))/2; % Step 0: 按等功率初始化分配

iter = 0; % 迭代计数
sum_rate = []; % 记录和速率

sum_rate_old = 100;

A = ones(U,C) - eye(U); % 求和指示矩阵

while(iter < Max_iter)
    iter = iter + 1;

    % Step 1 - 修复矩阵运算
    interference = H_gain .* A * p_temp + 1;
    y_star = sqrt(diag(H_gain) .* p_temp) ./ interference; % 公式（33）

    % Step 2 - 简化CVX表达式
    cvx_begin quiet
        variable p(C,1)
        
        % 分解复杂的表达式
        term1 = 2 * y_star .* sqrt(diag(H_gain) .* p);
        interference_term = H_gain .* A * p + 1;
        term2 = y_star.^2 .* interference_term;
        
        % 使用更稳定的表达式
        objective_terms = log(1 + term1 - term2);
        
        maximize(sum(objective_terms)) % 公式（32）
        subject to
            p >= 0;
            p <= db2pow(P);
    cvx_end

    if ~isnan(cvx_optval) && ~isempty(cvx_optval)
        sum_rate = [sum_rate cvx_optval];
        p_temp = p;
        
        if abs(cvx_optval - sum_rate_old) / sum_rate_old < epsilon
            break;
        else
            sum_rate_old = cvx_optval;
        end
    else
        warning('CVX optimization failed at iteration %d', iter);
        break;
    end
end

if ~isempty(sum_rate)
    figure;
    plot(1:length(sum_rate), sum_rate/log(2)*B/1e6,'-b*','LineWidth',1)
    grid on
    hold on
end

%闭式解方法
A = ones(U,C) - eye(U); % 求和指示矩阵

p = db2pow(P*ones(C,1))/2; % Step 0: 按等功率初始化分配
gamma = diag(H_gain) .* p ./ (sum(H_gain .* A .* p', 2) + 1); % 修复矩阵运算
y_star = sqrt((1+gamma) .* diag(H_gain) .* p) ./ (sum(H_gain .* p', 2) + 1);

iter = 0; % 迭代计数
sum_rate2 = []; % 记录和速率

sum_rate_old = 100;

while(iter < Max_iter)
    iter = iter + 1;

    % Step 1
    y_star = sqrt((1+gamma) .* diag(H_gain) .* p) ./ (sum(H_gain .* p', 2) + 1); % 公式（45）

    % Step 2
    a = y_star.^2 .* diag(H_gain) .* p;
    gamma = (a + sqrt(a.^2 + 4*a)) / 2;

    % Step 3 - 修复分母计算
    denominator = sum(y_star.^2 .* H_gain, 1)'.^2;
    p = min(db2pow(P), y_star.^2 .* (1+gamma) .* diag(H_gain) ./ denominator);

    opt_value = sum(log2(1 + diag(H_gain) .* p ./ (sum(H_gain .* A .* p', 2) + 1)));

    if ~isnan(opt_value)
        sum_rate2 = [sum_rate2 opt_value];
    end

    if abs(opt_value - sum_rate_old) / sum_rate_old < epsilon
        break;
    else
        sum_rate_old = opt_value;
    end
end

if ~isempty(sum_rate2)
    plot(1:length(sum_rate2), sum_rate2*B/1e6,'-ro','LineWidth',1)
    xlim([1 max(length(sum_rate), length(sum_rate2))])
    grid on
    xlabel('Iteration number')
    ylabel('Sum rate (Mbps)')
    legend('Direct FP','Close-form FP','Location','southeast')
end

function [H_gain] = channel_generate(U,R,PL,shadowing_std)
% 在蜂窝小区覆盖的范围内产生用户位置，
% 大尺度衰落根据路损和阴影衰落模型计算
% 小区间隔0.8 km，小区覆盖区域为正六边形
% 基站位于正六边形中心
% U -- 用户个数
% R -- 基站距离
% PL -- 路损模型
% shadowing_std -- 正态对数阴影衰落标准差

% 在正六边形蜂窝小区中撒点
cell_loc = [0,0;
    R*cos(pi/6),R*sin(pi/6);
    0,R;
    -R*cos(pi/6),R*sin(pi/6);
    -R*cos(pi/6),-R*sin(pi/6);
    0,-R;
    R*cos(pi/6),-R*sin(pi/6)]; % 基站坐标
C = 7; % 蜂窝小区个数

L = R*tan(pi/6); % 六边形的边长

% 产生用户位置
user_loc = zeros(U,2);
i=0;
while i<U
    x = 2*L*rand(1,2) - 1*L;
    if (abs(x(1)) + abs(x(2))/sqrt(3) ) <= L && abs(x(2)) <= L*sqrt(3)/2
        i=i+1;
        user_loc(i,:)=x+cell_loc(i,:);
    end
end

% 计算距离
dis = zeros(U,C);
for i=1:U
    for j=1:C
        dis(i,j) = norm(cell_loc(j,:)-user_loc(i,:));
    end
end

% 计算信道增益，考虑服从对数正态分布的阴影衰落
H_gain = -PL(dis/1000) - shadowing_std*randn(U,C); % 注意：距离单位转换为km
end