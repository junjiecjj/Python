clear; close all; clc;

Pfa0 = 10 ^ (-8);
noise_factor_dB = linspace(0, 12, 100);
N = 5;
noise_factor = db2pow(noise_factor_dB);

T_nor = gammaincinv(1 - Pfa0, N);
Pfa = 1 - gammainc(T_nor ./ noise_factor, N);
Pfa_factor = Pfa ./ Pfa0;
plot(noise_factor_dB, log10(Pfa_factor), 'k-', 'linewidth', 1);
hold on;
grid on;


Pfa0 = 10 ^ (-6);
T_nor = gammaincinv(1 - Pfa0, N);
Pfa = 1 - gammainc(T_nor ./ noise_factor, N);
Pfa_factor = Pfa ./ Pfa0;
plot(noise_factor_dB, log10(Pfa_factor), 'k--', 'linewidth', 1);


Pfa0 = 10 ^ (-4);
T_nor = gammaincinv(1 - Pfa0, N);
Pfa = 1 - gammainc(T_nor ./ noise_factor, N);
Pfa_factor = Pfa ./ Pfa0;
plot(noise_factor_dB, log10(Pfa_factor), 'k-.', 'linewidth', 1);


xlabel('\sigma_{w}^2 / \sigma_{w0}^2 /dB');
ylabel('log_{10}(P_{FA} / P_{FA0})');
title('噪声功率偏差的影响');
legend('P_{FA}=10^{-8}', 'P_{FA}=10^{-6}', 'P_{FA}=10^{-4}');

% clear; close all; clc;
% noise_factor_dB = linspace(0, 20, 100);
% noise_factor = db2pow(noise_factor_dB);
% Pfa0 = 10 ^ (-8);
% Pfa_factor = Pfa0 .^ (1 ./ noise_factor - 1);

% plot(noise_factor_dB, log10(Pfa_factor));