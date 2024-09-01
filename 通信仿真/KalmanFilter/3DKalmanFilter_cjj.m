
%% https://blog.csdn.net/weixin_41869763/article/details/104812479
% GPS精度为1m、气压计精度为0.5m，加速度计的精度为 1cm/s^2
% 无人机按照螺旋线飞行，半径为 20m，螺距为40m，100s完成一圈飞行
% 数据采集频率为 10Hz

clear all

D = 3;                          % 维度，可取 1,2,3

dt = 0.1;                       % 0.1s采集一次数据
t0 = 0:dt:100;                  % 0~100s
N = length(t0);                 % 采样点数

A = eye(D);                     % 状态转移矩阵，和上一时刻状态没有换算，故取 D阶单位矩阵
x = zeros(D, N);                % 存储滤波后的数据
z = ones(D, N);                 % 存储滤波前的数据
x(:, 1) = ones(D,1);            % 初始值设为 1（可为任意数）
P = eye(D);                     % 初始值为 1（可为非零任意数），取 D阶单位矩阵

r = 20;                         % 绕圈半径，20m
w = 2*pi / 100;                 % 计算出角速度，100s绕一圈

Q = 1e-2*eye(D);                % 过程噪声协方差，估计一个
R = [1 0 0;
     0 1 0;
     0 0 0.5];                  % 测量噪声协方差，精度为多少取多少

k = 1;                          % 采样点计数

if D==1                         % 一维仅高度，气压计数据
    true1D = t0*0.4;
elseif D==2                     % 二维 x,y 方向，GPS数据
    true2D = [r*cos(w*t0); r*sin(w*t0)];
elseif D==3                     % 三维 x,y,z方向，GPS和气压计
    true3D = [r * cos(w*t0); r * sin(w*t0); t0 * 0.4];
end

for t = dt:dt:100
    k = k + 1;
    x(:,k) = A * x(:,k-1);      % 卡尔曼公式1
    P = A * P * A' + Q;         % 卡尔曼公式2
    H = eye(D);
    K = P*H' * inv(H*P*H' + R); % 卡尔曼公式3
    if D==1                                     % 一维仅高度（z方向）
        z(:,k) = true1D(k) + randn;
    elseif D==2                                 % 二维 x,y 方向
        z(:,k) = [true2D(1,k) + randn; true2D(2,k) + randn];
    elseif D==3                                 % 三维 x,y,z 方向
        z(:,k) = [true3D(1,k) + randn; true3D(2,k) + randn; true3D(3,k) + randn];
    end
    x(:,k) = x(:,k) + K * (z(:,k)-H*x(:,k));    % 卡尔曼公式4
    P = (eye(D)-K*H) * P;                       % 卡尔曼公式5
end

if D==1                                         %% 一维情况
    plot(t0, z,'b.');                           % 绘制滤波前数据
    axis('equal');hold on;grid on;              % 坐标等距、继续绘图、添加网格
    plot(t0, x,'r.');                           % 绘制滤波后数据
    plot(t0, true1D ,'k-');                     % 绘制真实值
    legend('滤波前','滤波后','理想值');           % 标注
    xlabel('时间: s');
    ylabel('高度: m');hold off;
elseif D==2                                     %% 二维情况
    plot(z(1,:),z(2,:),'b.');                   % 绘制滤波前数据
    axis('equal');grid on;hold on;              % 坐标等距、继续绘图、添加网格
    plot(x(1,:),x(2,:),'r.');                   % 绘制滤波后数据
    plot(true2D(1,:), true2D(2,:), 'k.');       % 绘制真实值
    legend('滤波前','滤波后','理想值');
    xlabel('x方向: m');
    ylabel('y方向: m');hold off;
elseif D==3                                     %% 三维情况
    plot3(z(1,:),z(2,:),z(3,:),'b.');           % 绘制滤波前数据
    axis('equal');grid on;hold on               % 坐标等距、继续绘图、添加网格
    plot3(x(1,:),x(2,:),x(3,:),'r.');           % 绘制滤波后数据
    plot3(true3D(1,:), true3D(2,:), true3D(3,:));% 绘制滤波后数据
    legend('滤波前','滤波后','理想值');           % 绘制真实值
    xlabel('x方向: m');
    ylabel('y方向: m');
    zlabel('高度: m');hold off;
end


















