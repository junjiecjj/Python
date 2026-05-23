clc;
clear;
close all;

M = 10;
d_lambda = 0.5;
SNR_dB = 0;
SNR = 10^(SNR_dB / 10);

theta_target_deg = 0;
theta_beam_deg = linspace(0, 12, 1201);

n = (-(M - 1) / 2 : (M - 1) / 2).';

a_fun = @(theta) exp(-1j * 2 * pi * d_lambda * n * sin(theta));
ad_fun = @(theta) -1j * 2 * pi * d_lambda * cos(theta) * n .* a_fun(theta);

theta_target = deg2rad(theta_target_deg);

a0 = a_fun(theta_target);
ad0 = ad_fun(theta_target);

A0 = a0 * a0.';
Ad0 = ad0 * a0.' + a0 * ad0.';

CRB_coherent = zeros(size(theta_beam_deg));
CRB_orth = zeros(size(theta_beam_deg));

% 为了贴近原文 Fig. 7 的数值标定
SNR_coherent_eff = M * SNR;
SNR_orth_eff = M^2 * SNR;

Rs_orth = eye(M);
CRB_orth_value = crb_single_target_trace(A0, Ad0, Rs_orth, SNR_orth_eff);

for idx = 1:length(theta_beam_deg)
    theta_beam = deg2rad(theta_beam_deg(idx));

    ab = a_fun(theta_beam);

    % 相干发射波束指向 theta_beam
    % s_m[n] = u_m s[n], R_s = u u^H
    % 取 u = conj(a(theta_beam))，使发射波束指向 theta_beam
    u = conj(ab);
    Rs_coherent = u * u';

    CRB_coherent(idx) = crb_single_target_trace(A0, Ad0, Rs_coherent, SNR_coherent_eff);
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

function CRB = crb_single_target_trace(A, Ad, Rs, SNR)
    term_AA = trace(A * Rs * A');
    term_DD = trace(Ad * Rs * Ad');
    term_DA = trace(Ad * Rs * A');

    denominator = 2 * SNR * (term_DD * term_AA - abs(term_DA)^2);
    numerator = term_AA;

    CRB = real(numerator / denominator);
end