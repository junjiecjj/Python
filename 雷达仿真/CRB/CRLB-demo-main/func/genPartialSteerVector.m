function a = genPartialSteerVector(theta, N, d, lambda, flag)
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
if flag == 1    
    a = (1j*pi.*n*cos(asin(theta))).*exp(1j*2*pi*d*theta/lambda*n);
else
    a = -(pi.*n.*cos(asin(theta))).^2.*exp(1j*2*pi*d*theta/lambda*n) - (1j*pi.*n*theta).*exp(1j*2*pi*d*theta/lambda*n);
end
end