%% Fig. 9: Optimal and sample-MVDR output SINRs, N = 10
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
clear;
clc;
close all;

%% Parameters: Examples 5 and 6
M = 10;
N = 10;
K_list = [1, M, 5];
dT = 0.5;
dR = 0.5;
theta_s = 10;
theta_i = [-30, -10];
SNR_dB = -30:5:30;
INR_dB = 30;
sigma_n2 = 1;
sigma_i2 = sigma_n2 * 10^(INR_dB / 10);
sigma_s2 = sigma_n2 * 10.^(SNR_dB / 10);
N_snap = 100;
Iter = 100;
DL = sigma_n2;                    % Supplement: paper does not specify DL
rng(42, 'twister');               % Supplement: paper does not specify seed
gain_optimal = zeros(3, 1);
gain_MVDR = zeros(3, Iter);
u_s = cell(3, 1);
U_i = cell(3, 1);
R_i_n = cell(3, 1);

%% Known-covariance optimal SINR: Eqs. (41), (42), and (58)
for method = 1:3
    K = K_list(method);
    u_s{method} = virtual_array(M, N, K, dT, dR, theta_s, theta_s);
    U_i{method} = virtual_array(M, N, K, dT, dR, theta_s, theta_i);
    R_i_n{method} = (M / K) * sigma_i2 * (U_i{method} * U_i{method}') ...
        + sigma_n2 * eye(K * N);
    v = R_i_n{method} \ u_s{method};
    gain_optimal(method) = (M / K) * real(u_s{method}' * v);
end

%% Target-free training snapshots; evaluate SINR with the TRUE covariance
for iter = 1:Iter
    Z_i = (randn(2, N_snap) + 1j * randn(2, N_snap)) / sqrt(2);
    Z_n = (randn(M * N, N_snap) + 1j * randn(M * N, N_snap)) / sqrt(2);
    for method = 1:3
        K = K_list(method);
        P = K * N;
        Y = sqrt((M / K) * sigma_i2) * U_i{method} * Z_i ...
            + sqrt(sigma_n2) * Z_n(1:P, :);
        R_hat = (Y * Y') / N_snap;
        R_loaded = (R_hat + R_hat') / 2 + DL * eye(P);
        v = R_loaded \ u_s{method};
        w = v / (u_s{method}' * v);
        assert(abs(w' * u_s{method} - 1) < 1e-8, ...
            'Distortionless constraint failed.');
        gain_MVDR(method, iter) = (M / K) * abs(w' * u_s{method})^2 ...
            / real(w' * R_i_n{method} * w);
    end
end
assert(all(all(gain_MVDR <= repmat(gain_optimal, 1, Iter) ...
    * (1 + 1e-8))), 'Sample SINR exceeds the optimal bound.');
% Average linear SINRs before converting to dB: reproduction choice.
SINR_optimal = gain_optimal * sigma_s2;
SINR_MVDR = mean(gain_MVDR, 2) * sigma_s2;
optimal_dB = 10 * log10(SINR_optimal);
MVDR_dB = 10 * log10(SINR_MVDR);
fprintf('Mean sample-MVDR loss: %.6f, %.6f, %.6f dB\n', ...
    10 * log10(gain_optimal ./ mean(gain_MVDR, 2)));

%% Unified single-axes style; original line styles and marker types
width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

figure(9);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

plot(SNR_dB, MVDR_dB(1, :), 's--', 'Color', '#F65314', 'LineWidth', linewidth, 'MarkerSize', markersize);
hold on;
plot(SNR_dB, optimal_dB(1, :), 'd--', 'Color', '#00A1F1', 'LineWidth', linewidth, 'MarkerSize', markersize, 'MarkerFaceColor', '#00A1F1');
plot(SNR_dB, MVDR_dB(2, :), 'd-.', 'Color', '#8A2BE2', 'LineWidth', linewidth, 'MarkerSize', markersize);
plot(SNR_dB, optimal_dB(2, :), '.-.', 'Color', '#A9A9A9', 'LineWidth', linewidth, 'MarkerSize', markersize);
plot(SNR_dB, MVDR_dB(3, :), 'o-', 'Color', '#7EBA00', 'LineWidth', linewidth, 'MarkerSize', markersize);
plot(SNR_dB, optimal_dB(3, :), '.-.', 'Color', '#1F4E79', 'LineWidth', linewidth, 'MarkerSize', markersize);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend({'PHASED-ARRAY RADAR (MVDR)', 'PHASED-ARRAY RADAR (OPTIMAL)', ...
    'MIMO RADAR (MVDR)', 'MIMO RADAR (OPTIMAL)', ...
    'PHASED-MIMO RADAR (MVDR; K=5)', 'PHASED-MIMO RADAR (OPTIMAL; K=5)'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('SNR (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Output SINR (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-30, 30]);
ylim([-12, 70]);
xticks(-30:10:30);
yticks(-10:10:70);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig9.png','-dpng','-r600');
print(gcf,'Fig9.pdf','-dpdf','-vector');

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