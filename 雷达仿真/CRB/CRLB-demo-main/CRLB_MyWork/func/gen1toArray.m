function [h] = gen1toArray(xp, yp,...
                           x1, y1,...
                           xN, yN,...
                           N, fc)
%% 生成1点到一个阵列的信道                       
% Inputs:
%       xp,yp是点目标的位置
%       x1, y1是线阵的头坐标
%       xN, yN是线阵的尾坐标
%       N:线阵的阵元数
% Outputs:
%       h                       =       点目标到线阵的信道（N×1）
%%
lambda = 3e8/fc;
kc = 2*pi/lambda;
x = linspace(x1, xN, N).';% 生成线阵的x坐标
y = linspace(y1, yN, N).';% 生成线阵的y坐标
h = exp(-1j*kc* sqrt((x-xp).^2 + (y-yp).^2));

% [b] = genb(theta, r, N, fc);


end