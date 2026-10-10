%% Fig. 7: Nonadaptive SINR with uniformly distributed interference
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
clear;
clc;
close all;

%% Parameters: Example 4
M = 10;
N = 10;
K_list = [1, M, 5];
dT = 0.5;
dR = 0.5;
theta_s = 10;
theta_i = -50:0.02:-20;           % Quadrature grid: reproduction choice
INR_dB = -20:5:50;
SNR_dB = INR_dB;
sigma_n2 = 1;
sigma_i2 = sigma_n2 * 10.^(INR_dB / 10);
sigma_s2 = sigma_n2 * 10.^(SNR_dB / 10);
sector_width = theta_i(end) - theta_i(1);
SINR = zeros(3, numel(INR_dB));

%% Continuous angular covariance model and Eq. (41)
% Uniform power per degree; the integral is normalized by sector width.
for method = 1:3
    K = K_list(method);
    u_s = virtual_array(M, N, K, dT, dR, theta_s, theta_s);
    U_i = virtual_array(M, N, K, dT, dR, theta_s, theta_i);
    signal_gain = (M / K) * abs(u_s' * u_s)^2;
    interference_gain = (M / K) * trapz(theta_i, ...
        abs(u_s' * U_i).^2) / sector_width;
    noise_power = sigma_n2 * norm(u_s)^2;
    SINR(method, :) = sigma_s2 * signal_gain ...
        ./ (sigma_i2 * interference_gain + noise_power);
    fprintf('Method %d, high-INR limit: %.6f dB\n', ...
        method, 10 * log10(signal_gain / interference_gain));
end
assert(all(all(diff(SINR, 1, 2) > 0)), 'SINR monotonicity check failed.');
SINR_dB = 10 * log10(SINR);

%% Unified single-axes style; original line styles and marker types
width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

figure(7);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

plot(INR_dB, SINR_dB(1, :), 's--', 'Color', '#F65314', 'LineWidth', linewidth, 'MarkerSize', markersize);
hold on;
plot(INR_dB, SINR_dB(2, :), 'v-.', 'Color', '#00A1F1', 'LineWidth', linewidth, 'MarkerSize', markersize);
plot(INR_dB, SINR_dB(3, :), 'o-', 'Color', '#8A2BE2', 'LineWidth', linewidth, 'MarkerSize', markersize);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend({'PHASED-ARRAY RADAR', 'MIMO RADAR', 'PHASED-MIMO RADAR (K=5)'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('INR (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Output SINR (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-20, 50]);
ylim([0, 55]);
xticks(-20:10:50);
yticks(0:5:55);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig7.png','-dpng','-r600');
print(gcf,'Fig7.pdf','-dpdf','-vector');

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