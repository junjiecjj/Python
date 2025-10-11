% ------- һ��ǰ��Ԥ���㷨 -------------------------
%  DOA estimation
% Developed by xiaofei zhang (�Ͼ����պ����ѧ ���ӹ���ϵ ��С�ɣ�
% EMAIL:zhangxiaofei@nuaa.edu.cn

clear all;
clc;
close all;

wavelength = 1;    % ������һ��
d = 0.5 * wavelength;     % ��Ԫ��ľ���

% ----- Ӱ��ֱ�������Ҫ���� ---
element_num  = 10;        % ��Ԫ�����Էֱ��������������
snapshot_num = 64;       % ������
snr          = 10;        % �������ߵ��ο��������
doa          = [60 80, 90 100];   % �źŷ�����������ж���ź�

% -------------------
num_s = length(doa);   % �ź���
fd  = linspace(0, 2000, num_s);
prf = 5000;
signal_power = 1;      % �����źŹ��ʣ�������Ϊ�ȹ��ʣ�ʵ���Ϲ��ʲ��Ҳ��Ӱ��ֱ��������ﲻ���ǹ��ʲ�
step = 0.04;           % ����ֵ��Ϊ�˸���ϸ�Ĺ��ƿ��Ե�СЩ

for noTrial = 1:10     % 10�η���
  % �����ź�
  signal = receive_signal(doa, d, element_num, wavelength, fd, prf, signal_power, snr, snapshot_num);

  X = signal;
  R = X*X'./snapshot_num;         % ����Э�������
 
  % ------- һ��ǰ��Ԥ���㷨 -------------------------
  Xf   = flipud(X(1:element_num-1, :)); 
  Rf   = Xf*Xf'/snapshot_num;
  rf   = Xf*X(element_num, :)'/snapshot_num;
  Wflp = conj((inv(Rf)*rf));

  theta = 0 : step : 180;
  for k = 1 : length(theta);
      a = steering_vector(theta(k), wavelength, d, element_num); 
      Pflp(noTrial, k) = abs(1 ./ (a'*[1;-Wflp]));  % �ռ���
  end

end

Pflpm = mean(Pflp);

%%%%% ��ͼ
figure;
plot(theta, Pflpm);  
grid on;
xlabel('�Ƕȣ�/\circ)');
ylabel('�ռ���');
title('DOA estimation');








% -------------------------------------------------------------
