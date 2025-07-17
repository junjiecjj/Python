function a = genSteerVector(theta, N, d, lambda)
%%  生成导向矢量,theta是虚拟角度

% inputs
%       theta                   =       虚拟角度
%       N                       =       天线阵元数
%       d                       =       天线阵元间隔
%       lambda                  =       波长

% outputs
%       a                       =       生成的导向矢量（N×1）
%%
% n = ((2*(0:N-1)-N+1)/2).';% 戴老师生成方式，角度按中间的阵元算
n = (0:(N-1)).';
a = exp(1j*2*pi*d*theta/lambda*n);

end