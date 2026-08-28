%% Fig. 6(b)
% Uncovering the Iceberg in the Sea: Fundamentals of Pulse Shaping and
% Modulation Design for Random ISAC Signals
%
% OFDM, 16-QAM, N = 128, L = 10, alpha = 0.35. Unlike Fig. 6(a), this
% panel uses the original unoptimized RRC pulse.

clear;
close all;
clc;
rng(42,'twister');

set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultTextFontSize',18);
set(groot,'defaultLineLineWidth',2);
set(groot,'defaultLineMarkerSize',6);
set(groot,'defaultFigureColor','white');
set(groot,'defaultLegendFontSize',14);

%% Parameters in Fig. 6(b)
Tsym = 1;
pi_value = pi;
N = 128;
L = 10;
alpha = 0.35;
kappa = 1.32;

FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);

%% Generate the original, unoptimized RRC pulse
[p,t,filtDelay] = commpyRrcosfilter(L*N,alpha,Tsym,L/Tsym); 
p = p/sqrt(sum(abs(p).^2));

norm2p = norm(p); 
g = N*(FLN*p).*(conj(FLN)*conj(p));
g = real(g);

%% OFDM theoretical average squared ACF, Eq. (36)
M = 10000;
U = FN'; 
V = eye(N);
tilde_V = V.*conj(V); 

TheoAveACF_Iceberg = zeros(L*N,1);
TheoAveACF_OFDM_M1 = zeros(L*N,1);
TheoAveACF_OFDM_M10000 = zeros(L*N,1);

for k = 0:L*N-1
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);
    fk = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));

    r1 = abs(gk.'*conj(fk))^2;
    TheoAveACF_Iceberg(k+1) = r1;

    M = 1;
    r2 = (kappa-1)/M* (N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_M1(k+1) = r1+r2;

    M = 1000;
    r2 = (kappa-1)/M* (N-2*(1-cos(2*pi_value*k/L))*sum(g(1:N).*(1-g(1:N))));
    TheoAveACF_OFDM_M10000(k+1) = r1+r2;
end

TheoAveACF_Iceberg = TheoAveACF_Iceberg/max(TheoAveACF_Iceberg)+1e-14;
TheoAveACF_Iceberg = fftshift(TheoAveACF_Iceberg);

TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1/max(TheoAveACF_OFDM_M1)+1e-14;
TheoAveACF_OFDM_M1 = fftshift(TheoAveACF_OFDM_M1);

TheoAveACF_OFDM_M10000 = TheoAveACF_OFDM_M10000/max(TheoAveACF_OFDM_M10000)+1e-14;
TheoAveACF_OFDM_M10000 = fftshift(TheoAveACF_OFDM_M10000);

%% Numerical average squared ACF, inherited from Fig2.py Eq. (26)
MOD_TYPE = 'qam';  
Order = 16;
Constellation = qammod((0:Order-1).',Order,'gray','UnitAveragePower',true);
AvgEnergy = mean(abs(Constellation).^2); 

% Preserve the original full three-dimensional array and loop order.
Iter = 100;

%% No coherent integration: M = 1
M = 1;
SimAveACF_OFDM_M1 = zeros(Iter,L*N);

for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);

    for it = 1:Iter
        d = randi([0,Order-1],N,1);
        s = Constellation(d+1);
        VHs = abs(V'*s).^2;
        SimAveACF_OFDM_M1(it,k+1) = abs(sum(gk.*VHs.*conj(fk_tilde)))^2;
    end
end

Sim_M1_avg = mean(SimAveACF_OFDM_M1,1);
Sim_M1_avg = Sim_M1_avg/max(Sim_M1_avg)+1e-14;
Sim_M1_avg = fftshift(Sim_M1_avg);

%% 10,000 coherent integrations: M = 10000
M = 1000;
SimAveACF_OFDM_M10000 = complex(zeros(M,Iter,L*N));

parfor k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);

    for m = 1:M
        for it = 1:Iter
            d = randi([0,Order-1],N,1);
            s = Constellation(d+1);
            VHs = abs(V'*s).^2;

            % First retain the complex ACF; do not square before averaging.
            SimAveACF_OFDM_M10000(m,it,k+1) = ...
                sum(gk.*VHs.*conj(fk_tilde));
        end
    end
end

%% Coherent averaging, Eq. (33)
RkBar = mean(SimAveACF_OFDM_M10000,1);
RkBar2 = abs(RkBar).^2;

Sim_M10000_avg = squeeze(mean(RkBar2,2)).';
Sim_M10000_avg = Sim_M10000_avg/max(Sim_M10000_avg)+1e-14;
Sim_M10000_avg = fftshift(Sim_M10000_avg);


%% Plot Fig. 6(b)

%% ===========================================
width = 8;%设置图宽
height = 6;%设置图高
fontsize = 14;%设置图中字体大小
linewidth = 2;%设置线宽
markersize = 10;%标记大小
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
% ===========================================

figure(1);
set(gcf, 'Units', 'inches');
% set(gcf, 'Position', [0, 0, width, height]);
set(gcf, 'Color', 'white');
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);

x = -N*L/2:N*L/2-1;
plot(x,10*log10(Sim_M1_avg),'-', 'Color','#F65314', 'LineWidth',1.5); hold on;
plot(x,10*log10(TheoAveACF_OFDM_M1),'--', 'Color','#05C349', 'LineWidth',1.5);
plot(x,10*log10(Sim_M10000_avg),'-', 'Color','#00A1F1', 'LineWidth',1.5);
plot(x,10*log10(TheoAveACF_OFDM_M10000),'--', 'Color','#8A2BE2', 'LineWidth',1.5);
plot(x,10*log10(TheoAveACF_Iceberg),'--', 'Color','#B0B0B0', 'LineWidth',2);

set(gca, 'FontSize',16, 'FontName','Times New Roman');
h_legend = legend('No Integration, Numerical', 'No Integration, Theoretical', '1k Coh Integration, Numerical', '1k Coh Integration, Theoretical', '``Iceberg" of the RRC Pulse', 'Interpreter','latex');
legendsize = 13;
set(h_legend, 'FontName','Times New Roman', 'FontSize',legendsize, 'FontWeight','normal', 'LineWidth',1, 'Location','northwest');

labelsize = 16;
xlabel('Delay Index', 'FontSize',labelsize, 'FontName','Times New Roman', 'Interpreter','latex');
ylabel('Ambiguity Level (dB)', 'FontSize',labelsize, 'FontName','Times New Roman', 'Interpreter','latex');

xlim([-300,300]);
ylim([-120,0]);
xticks(-300:100:300);
yticks(-120:20:0);

grid on;
set(gca, 'GridLineStyle','--', 'GridAlpha',0.2, 'LineWidth',1, 'GridLineWidth',0.5, 'Layer','bottom');

set(gca, 'Units','normalized');
set(gca, 'Position',[0.125,0.125,0.85,0.86]);

% print(gcf, './Figs/Fig_6b_m.pdf', '-dpdf', '-vector');

function mat = FFTmatrix(L)
mat = complex(zeros(L,L));
for i = 0:L-1
    for j = 0:L-1
        mat(i+1,j+1) = exp(-1j*2*pi*i*j/L)/sqrt(L);
    end
end
end


function [p,t,filtDelay] = commpyRrcosfilter(N,alpha,Ts,Fs)
% MATLAB translation of commpy.filters.rrcosfilter used by Fig2.py.
T_delta = 1/Fs;
t = ((0:N-1).'-N/2)*T_delta;
p = zeros(N,1);

for x = 1:N
    t_x = t(x);

    if t_x == 0
        p(x) = 1-alpha+4*alpha/pi;
    elseif alpha ~= 0 && t_x == Ts/(4*alpha)
        p(x) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha)) + ...
            (1-2/pi)*cos(pi/(4*alpha)));
    elseif alpha ~= 0 && t_x == -Ts/(4*alpha)
        p(x) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha)) + ...
            (1-2/pi)*cos(pi/(4*alpha)));
    else
        p(x) = (sin(pi*t_x*(1-alpha)/Ts) + ...
            4*alpha*(t_x/Ts)*cos(pi*t_x*(1+alpha)/Ts)) / ...
            (pi*t_x*(1-(4*alpha*t_x/Ts)^2)/Ts);
    end
end

filtDelay = (N-1)/2;
end
