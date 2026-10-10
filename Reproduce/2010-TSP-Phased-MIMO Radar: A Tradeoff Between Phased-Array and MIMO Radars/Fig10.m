%% Fig. 10: Overall MVDR beampatterns, N = 1
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
clear;
clc;
close all;

%% Parameters: Examples 5 and 6
M = 10;
N = 1;
K_list = [1, M, 5];
dT = 0.5;
dR = 0.5;
theta_s = 10;
theta_i = [-30, -10];
theta = -90:0.1:90;
INR_dB = 50;
sigma_n2 = 1;
sigma_i2 = sigma_n2 * 10^(INR_dB / 10);
N_snap = 100;
DL = sigma_n2;                    % Supplement: paper does not specify DL
use_sample_covariance = true;
rng(42, 'twister');               % Supplement: paper does not specify seed
G = zeros(3, numel(theta));
Z_i = (randn(2, N_snap) + 1j * randn(2, N_snap)) / sqrt(2);
Z_n = (randn(M * N, N_snap) + 1j * randn(M * N, N_snap)) / sqrt(2);

%% One seeded training realization; pattern averaging is not specified
for method = 1:3
    K = K_list(method);
    U = virtual_array(M, N, K, dT, dR, theta_s, theta);
    u_s = virtual_array(M, N, K, dT, dR, theta_s, theta_s);
    U_i = virtual_array(M, N, K, dT, dR, theta_s, theta_i);
    P = K * N;
    R_i_n = (M / K) * sigma_i2 * (U_i * U_i') + sigma_n2 * eye(P);
    if use_sample_covariance
        Y = sqrt((M / K) * sigma_i2) * U_i * Z_i ...
            + sqrt(sigma_n2) * Z_n(1:P, :);
        R_hat = (Y * Y') / N_snap;
        R_used = (R_hat + R_hat') / 2 + DL * eye(P);
    else
        R_used = R_i_n;            % Known-covariance diagnostic option
    end
    v = R_used \ u_s;
    w = v / (u_s' * v);
    assert(abs(w' * u_s - 1) < 1e-8, 'Distortionless constraint failed.');
    G(method, :) = abs(w' * U).^2 / abs(w' * u_s)^2;
    fprintf('Method %d, response at interferers: %.3f, %.3f dB\n', ...
        method, 10 * log10(max(abs(w' * U_i).^2, realmin)));
end
G_dB = 10 * log10(max(G, realmin));

%% Unified single-axes style; original line styles and marker types
width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

figure(10);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

plot(theta, G_dB(1, :), '--', 'Color', '#F65314', 'LineWidth', linewidth);
hold on;
plot(theta, G_dB(2, :), '-.', 'Color', '#00A1F1', 'LineWidth', linewidth);
plot(theta, G_dB(3, :), '-', 'Color', '#8A2BE2', 'LineWidth', linewidth);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend({'PHASED-ARRAY RADAR', 'MIMO RADAR', 'PHASED-MIMO RADAR (K=5)'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('ANGLE (DEGREES)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('$|G(\theta)|^2$ (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-90, 90]);
ylim([-120, 30]);
xticks(-80:20:80);
yticks(-120:20:20);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig10.png','-dpng','-r600');
print(gcf,'Fig10.pdf','-dpdf','-vector');

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