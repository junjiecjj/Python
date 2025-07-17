function [b] = genb(theta, r, N, f, d)
%% 生成近场导向矢量

% Inputs：
%       theta & r=          =       公式7输入参数
%       N                   =       天线数
%       fc                  =       载波频率
%       d                   =       天线阵元间距

% Outputs：
%       b                   =       近场导向矢量
%%
% b = zeros(N, 1);
kc = 2*pi*f/3e8;
delta_n = ((2*(0:N-1)-N+1)/2).';% 戴老师生成方式，角度按中间的阵元算
% delta_n = (0:N-1).';% 我使用的方式，角度按第一个阵元算
% d = lambda/2;
r_l_n = sqrt(r^2 + delta_n.^2.*d.^2 - 2*r*theta*delta_n*d);

b = exp(-1j*kc*(r_l_n-r))/sqrt(N);
end