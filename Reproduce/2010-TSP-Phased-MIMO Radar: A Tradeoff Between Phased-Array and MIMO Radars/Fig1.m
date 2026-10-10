%% Fig. 1: Transmit beampatterns using conventional beamformer
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
% C_K(theta): transmit factor defined immediately after Eq. (34).
% Fig. 1 contains C_K only, without D_K or the receive factor R.
clear;
clc;
close all;

%% Parameters: Section V and Example 1
M = 10;
K = 5;
dT = 0.5;                         % Transmit spacing in wavelengths
theta_s = 10;                     % Target direction in degrees
theta = -90:0.1:90;               % Reproduction choice; not specified
Mk = M - K + 1;                   % Six antennas per overlapping subarray

%% Full transmit steering vector: Eq. (8), negative exponent
m = (0:M - 1).';
a = exp(-1j * 2 * pi * dT * m * sind(theta));
a_s = exp(-1j * 2 * pi * dT * m * sind(theta_s));

%% Phased-array radar: K = 1, Eqs. (25), (34), and (36)
w_PH = a_s / norm(a_s);
C_PH = abs(w_PH' * a).^2 / abs(w_PH' * a_s).^2;

%% MIMO radar: K = M, one antenna per subarray, Eq. (37)
C_MIMO = ones(size(theta));

%% Phased-MIMO radar: fully overlapping subarrays, Eqs. (24)-(25)
% Subarrays: [1:6], [2:7], [3:8], [4:9], and [5:10].
W = zeros(M, K);
C_subarray = zeros(K, numel(theta));

for k = 1:K
    index = k:k + Mk - 1;
    mk = (0:Mk - 1).';
    a_k = exp(-1j * 2 * pi * dT * mk * sind(theta));
    a_k_s = exp(-1j * 2 * pi * dT * mk * sind(theta_s));
    w_k = a_k_s / norm(a_k_s);
    W(index, k) = w_k;
    C_subarray(k, :) = abs(w_k' * a_k).^2 ...
        / abs(w_k' * a_k_s).^2;
end

% Every subarray has the same normalized transmit beampattern.
C_PH_MIMO = C_subarray(1, :);

%% Independent checks: Eq. (34) and total pulse energy in Eq. (11)
delta = sind(theta) - sind(theta_s);
C_PH_check = abs(sum(exp(-1j * 2 * pi * dT ...
    * (0:M - 1).' * delta), 1) / M).^2;
C_PH_MIMO_check = abs(sum(exp(-1j * 2 * pi * dT ...
    * (0:Mk - 1).' * delta), 1) / Mk).^2;

err_PH = max(abs(C_PH - C_PH_check));
err_PH_MIMO = max(abs(C_PH_MIMO - C_PH_MIMO_check));
err_subarray = max(max(abs(C_subarray ...
    - repmat(C_PH_MIMO, K, 1))));
E_total = (M / K) * sum(abs(W(:)).^2);
[~, index_s] = min(abs(theta - theta_s));

assert(err_PH < 1e-12, 'Phased-array formula check failed.');
assert(err_PH_MIMO < 1e-12, 'Phased-MIMO formula check failed.');
assert(err_subarray < 1e-12, 'Subarray patterns are inconsistent.');
assert(abs(E_total - M) < 1e-12, 'Total pulse energy check failed.');
assert(abs(C_PH(index_s) - 1) < 1e-12, 'Phased-array peak check failed.');
assert(abs(C_PH_MIMO(index_s) - 1) < 1e-12, ...
    'Phased-MIMO peak check failed.');

%% Convert normalized power to dB; do not square C_K again
C_PH_dB = 10 * log10(max(C_PH, realmin));
C_MIMO_dB = 10 * log10(max(C_MIMO, realmin));
C_PH_MIMO_dB = 10 * log10(max(C_PH_MIMO, realmin));

%% Unified MATLAB single-axis plotting style
width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

figure(1);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

%% Original line styles: dashed, dotted, and solid; no markers
plot(theta,C_PH_dB,'--','Color','#F65314','LineWidth',linewidth);
hold on;
plot(theta,C_MIMO_dB,':','Color','#00A1F1','LineWidth',linewidth);
plot(theta,C_PH_MIMO_dB,'-','Color','#8A2BE2','LineWidth',linewidth);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend('PHASED-ARRAY RADAR','MIMO RADAR', ...
    'PHASED-MIMO RADAR (K=5)','Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize, ...
    'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('ANGLE (DEGREES)','FontSize',labelsize, ...
    'FontName','Times New Roman','Interpreter','latex');
ylabel('$|C(\theta)|^2$ (dB)','FontSize',labelsize, ...
    'FontName','Times New Roman','Interpreter','latex');
xlim([-90,90]);
ylim([-80,20]);
xticks(-80:20:80);
yticks(-80:10:20);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1, ...
    'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;

%% Save in the current MATLAB directory
print(gcf,'Fig1.png','-dpng','-r600');
print(gcf,'Fig1.pdf','-dpdf','-vector');

%% Numerical verification report
fprintf('Phased-array formula error: %.3e\n', err_PH);
fprintf('Phased-MIMO formula error: %.3e\n', err_PH_MIMO);
fprintf('Subarray consistency error: %.3e\n', err_subarray);
fprintf('Total transmitted pulse energy: %.12f (expected M = %d)\n', ...
    E_total, M);
fprintf('Normalized peaks at theta_s = %.1f deg: %.12f, %.12f\n', ...
    theta_s, C_PH(index_s), C_PH_MIMO(index_s));