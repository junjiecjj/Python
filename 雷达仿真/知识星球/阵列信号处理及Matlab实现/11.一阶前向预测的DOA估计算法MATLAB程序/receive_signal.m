function signal = receive_signal(phi, d, element_num, wavelength,fd, prf, signal_power, snr, snapshot_num)
% �źŲ���
%
A = array_manifold(phi, d, element_num, wavelength);
noise_power = signal_power / 10 .^ (snr / 10);
signal_num  = length(phi);
noise = wgn(element_num, snapshot_num, noise_power, 'linear', 'complex');
S = zeros(signal_num, snapshot_num);
theta = 2 *pi * rand(1, signal_num);
for i = 1 : signal_num
    w = 2 * pi * fd(i) / prf * (0 : snapshot_num - 1);
    S(i, :) = signal_power .^0.5 * exp(j * theta(i)) * exp(j * w); % ģ����ź�Ϊ���Ҳ�
end

signal = A * S + noise;
