%% Fig. 6: Nonadaptive output SINR, INR = -30 dB
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
clear;
clc;
close all;

%% Parameters: Example 3
M = 10;
N = 10;
K_list = [1, M, 5];
dT = 0.5;
dR = 0.5;
theta_s = 10;
theta_i = [-30, -10];
SNR_dB = -30:5:30;
INR_dB = -30;
sigma_n2 = 1;
sigma_s2 = sigma_n2 * 10.^(SNR_dB / 10);
sigma_i2 = sigma_n2 * 10^(INR_dB / 10);
SINR = zeros(3, numel(SNR_dB));

%% Exact nonadaptive SINR: Eqs. (41)-(45)
% Deterministic weights and known covariance give the ensemble SINR directly.
for method = 1:3
    K = K_list(method);
    u_s = virtual_array(M, N, K, dT, dR, theta_s, theta_s);
    U_i = virtual_array(M, N, K, dT, dR, theta_s, theta_i);
    R_i_n = (M / K) * sigma_i2 * (U_i * U_i') ...
        + sigma_n2 * eye(K * N);
    w = u_s;
    signal_gain = (M / K) * abs(w' * u_s)^2;
    interference_noise = real(w' * R_i_n * w);
    denominator_check = (M / K) * sigma_i2 * sum(abs(w' * U_i).^2) ...
        + sigma_n2 * norm(w)^2;
    assert(abs(interference_noise - denominator_check) ...
        < 1e-10 * denominator_check, 'Covariance check failed.');
    SINR(method, :) = sigma_s2 * signal_gain / interference_noise;
end
SINR_dB = 10 * log10(SINR);
fprintf('Output SINR at SNR = 0 dB: %.6f, %.6f, %.6f dB\n', ...
    SINR_dB(:, SNR_dB == 0));

%% Unified single-axes style; original line styles and marker types
width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

figure(6);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

plot(SNR_dB, SINR_dB(1, :), 's--', 'Color', '#F65314', 'LineWidth', linewidth, 'MarkerSize', markersize);
hold on;
plot(SNR_dB, SINR_dB(2, :), 'v-.', 'Color', '#00A1F1', 'LineWidth', linewidth, 'MarkerSize', markersize);
plot(SNR_dB, SINR_dB(3, :), 'o-', 'Color', '#8A2BE2', 'LineWidth', linewidth, 'MarkerSize', markersize);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend({'PHASED-ARRAY RADAR', 'MIMO RADAR', 'PHASED-MIMO RADAR (K=5)'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('SNR (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Output SINR (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-30, 30]);
ylim([-10, 60]);
xticks(-30:10:30);
yticks(-10:10:60);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig6.png','-dpng','-r600');
print(gcf,'Fig6.pdf','-dpdf','-vector');

%% Virtual steering vector: Eqs. (13), (14), (18), and (25)
function U = virtual_array(M, N, K, dT, dR, theta_s, theta)
    Mk = M - K + 1;
    a = exp(-1j * 2 * pi * dT * (0:Mk - 1).' * sind(theta));
    a_s = exp(-1j * 2 * pi * dT * (0:Mk - 1).' * sind(theta_s));
    w = a_s / norm(a_s);
    c = w' * a;
    d = exp(-1j * 2 * pi * dT * (0:K - 1).' * sind(theta));
    b = exp(-1j * 2 * pi * dR * (0:N - 1).' * sind(theta));
    U = zeros(K * N, numel(theta));
    for k = 1:K
        index = (k - 1) * N + (1:N);
        U(index, :) = b .* repmat(c .* d(k, :), N, 1);
    end
end