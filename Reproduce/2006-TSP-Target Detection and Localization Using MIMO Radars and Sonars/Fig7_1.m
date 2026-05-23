clc;
clear;
close all;

M = 10;
d_lambda = 0.5;
SNR_dB = 0;
SNR_base = 10^(SNR_dB / 10);

theta_target_deg = 0;
theta_beam_deg = linspace(0, 12, 1201);

n = (-(M - 1) / 2 : (M - 1) / 2).';

a_fun = @(theta) exp(-1j * 2 * pi * d_lambda * n * sin(theta));
adot_fun = @(theta) -1j * 2 * pi * d_lambda * cos(theta) * n .* a_fun(theta);

theta_target = deg2rad(theta_target_deg);

a0 = a_fun(theta_target);
adot0 = adot_fun(theta_target);

CRB_coherent = zeros(size(theta_beam_deg));
CRB_orth = zeros(size(theta_beam_deg));

% 为了贴近原文 Fig. 7 的纵轴数值尺度
% 如果只想严格使用同一个 SNR_base，可把下面两行都改成 SNR_base
SNR_coherent = M * SNR_base;
SNR_orth = M^2 * SNR_base;

Rs_orth = eye(M);

CRB_orth_value = crb_single_target_correct67(a0, adot0, Rs_orth, SNR_orth);

for idx = 1:length(theta_beam_deg)
    theta_beam = deg2rad(theta_beam_deg(idx));

    a_beam = a_fun(theta_beam);

    w = conj(a_beam);

    Rs_coherent = w * w';

    CRB_coherent(idx) = crb_single_target_correct67(a0, adot0, Rs_coherent, SNR_coherent);
    CRB_orth(idx) = CRB_orth_value;
end

RMSE_coherent_deg = sqrt(CRB_coherent) * 180 / pi;
RMSE_orth_deg = sqrt(CRB_orth) * 180 / pi;

figure;
semilogy(theta_beam_deg, RMSE_coherent_deg, 'k--', 'LineWidth', 2);
hold on;
semilogy(theta_beam_deg, RMSE_orth_deg, 'k-', 'LineWidth', 2);

xlabel('\theta [deg]');
ylabel('RMSE [deg]');
legend('\beta = 1', '\beta = 0', 'Location', 'northwest');

grid on;
grid minor;
xlim([0, 12]);
ylim([2e-2, 1e2]);

function CRB = crb_single_target_correct67(a, adot, Rs, SNR)

    M = length(a);

    term1 = (a' * Rs.' * a) * (adot' * adot);
    term2 = M * (adot' * Rs.' * adot);
    term3 = M * abs(adot' * Rs.' * a)^2 / (a' * Rs.' * a);

    info = real(term1 + term2 - term3);

    CRB = 1 / (2 * SNR * info);

end