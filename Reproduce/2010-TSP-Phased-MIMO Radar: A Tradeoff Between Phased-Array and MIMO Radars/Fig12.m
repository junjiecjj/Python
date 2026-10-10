%% Fig. 12: Linear amplitude H_K, Appendix A, Eqs. (61)-(63)
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
clear;
clc;
close all;

%% Parameters: Fig. 12 and Appendix A
M = 10;
K_list = [1, 3, 4, 5];
dT = 0.5;
theta_s = 0;                      % Fig. 12 is centered at broadside
theta = -90:0.1:90;
Omega = 2 * pi * dT * (sind(theta) - sind(theta_s));
H = zeros(numel(K_list), numel(theta));

%% Eq. (63): amplitude product, not power and not dB
% The paper's sinc(kappa*Omega) means sin(kappa*Omega/2)/sin(Omega/2).
% Do not replace it with MATLAB's normalized sinc function.
regular = abs(sin(Omega / 2)) > 1e-12;
for method = 1:numel(K_list)
    K = K_list(method);
    Mk = M - K + 1;
    A = ones(size(theta));
    D = ones(size(theta));
    A(regular) = abs(sin(Mk * Omega(regular) / 2) ...
        ./ (Mk * sin(Omega(regular) / 2)));
    D(regular) = abs(sin(K * Omega(regular) / 2) ...
        ./ (K * sin(Omega(regular) / 2)));
    H(method, :) = A .* D;
    A_check = abs(sum(exp(-1j * (0:Mk - 1).' * Omega), 1)) / Mk;
    D_check = abs(sum(exp(-1j * (0:K - 1).' * Omega), 1)) / K;
    assert(max(abs(H(method, :) - A_check .* D_check)) < 1e-12, ...
        'Amplitude formula check failed.');
end
[~, index_s] = min(abs(theta - theta_s));
assert(all(abs(H(:, index_s) - 1) < 1e-12), 'Peak check failed.');

%% Unified single-axes style; original line styles and marker types
width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

figure(12);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

plot(theta, H(1, :), '--', 'Color', '#F65314', 'LineWidth', linewidth);
hold on;
plot(theta, H(2, :), ':', 'Color', '#00A1F1', 'LineWidth', linewidth);
plot(theta, H(3, :), '-.', 'Color', '#8A2BE2', 'LineWidth', linewidth);
plot(theta, H(4, :), '-', 'Color', '#A9A9A9', 'LineWidth', linewidth);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend({'PHASED-ARRAY (K=1)', 'PHASED-MIMO (K=3)', 'PHASED-MIMO (K=4)', 'PHASED-MIMO (K=5)'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('$\theta$ (DEGREES)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('$|H_K(\theta)|$','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-90, 90]);
ylim([0, 1.02]);
xticks(-80:20:80);
yticks(0:0.1:1);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig12.png','-dpng','-r600');
print(gcf,'Fig12.pdf','-dpdf','-vector');