% ------- 一阶前向预测算法 -------------------------
%  DOA estimation
% Developed by xiaofei zhang (南京航空航天大学 电子工程系 张小飞）
% EMAIL:zhangxiaofei@nuaa.edu.cn

clear all;
clc;
close all;

wavelength = 1;    % 波长归一化
d = 0.5 * wavelength;     % 阵元间的距离

% ----- 影响分辨力的重要参数 ---
element_num  = 10;        % 阵元数，对分辨率起决定性作用
snapshot_num = 64;       % 快拍数
snr          = 10;        % 单个天线单次快拍信噪比
doa          = [60 80, 90 100];   % 信号方向，这里可以有多个信号

% -------------------
num_s = length(doa);   % 信号数
fd  = linspace(0, 2000, num_s);
prf = 5000;
signal_power = 1;      % 单个信号功率，这里设为等功率，实际上功率差别也会影响分辨力，这里不考虑功率差
step = 0.04;           % 步进值，为了更精细的估计可以调小些

for noTrial = 1:10     % 10次仿真
  % 产生信号
  signal = receive_signal(doa, d, element_num, wavelength, fd, prf, signal_power, snr, snapshot_num);

  X = signal;
  R = X*X'./snapshot_num;         % 计算协方差矩阵
 
  % ------- 一阶前向预测算法 -------------------------
  Xf   = flipud(X(1:element_num-1, :)); 
  Rf   = Xf*Xf'/snapshot_num;
  rf   = Xf*X(element_num, :)'/snapshot_num;
  Wflp = conj((inv(Rf)*rf));

  theta = 0 : step : 180;
  for k = 1 : length(theta);
      a = steering_vector(theta(k), wavelength, d, element_num); 
      Pflp(noTrial, k) = abs(1 ./ (a'*[1;-Wflp]));  % 空间谱
  end

end

Pflpm = mean(Pflp);

%%%%% 画图
figure;
plot(theta, Pflpm);  
grid on;
xlabel('角度（/\circ)');
ylabel('空间谱');
title('DOA estimation');








% -------------------------------------------------------------
