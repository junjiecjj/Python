function H = build_delay_doppler_channel(L, delays, dopplers, gains)
%BUILD_DELAY_DOPPLER_CHANNEL 构造离散循环时延-多普勒信道矩阵
%
% y = Hx
% H = sum_p alpha_p * D(nu_p) * S(tau_p)
% 其中：
%   S(tau) 对应循环时延
%   D(nu)  对应离散多普勒调制

n = (0:L-1).';
I = eye(L);
H = zeros(L, L);

for p = 1:numel(gains)
    S_tau = circshift(I, delays(p), 1);
    D_nu  = diag(exp(1j * 2*pi * dopplers(p) * n / L));
    H = H + gains(p) * D_nu * S_tau;
end

end
