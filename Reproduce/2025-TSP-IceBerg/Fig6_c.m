%% Fig. 6(c)
% Direct comparison between the designed pulse and the RRC pulse

clear;
close all;
clc;
rng(42,'twister');

%% Parameters
Tsym = 1;
pi_value = pi;
N = 128;
L = 10;
alpha = 0.35;
kappa = 1.32;
M = 10000;
runNumericalSimulation = false;% false: only plot theoretical curves; true: also plot numerical curves

FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);

%% Generate the designed pulse according to (44)-(49)
K_s1 = 5*L:15*L;
gN = solve_iceberg_shaping_psl(N,L,alpha,K_s1);
g_N = 1-gN;
g_design = [gN;zeros((L-2)*N,1);g_N];
P_spectrum = sqrt(max(real(g_design),0)/N);
p_Designed = FLN'*P_spectrum;
p_Designed = p_Designed/sqrt(sum(abs(p_Designed).^2));
g_Designed = N*(FLN*p_Designed).*(conj(FLN)*conj(p_Designed));
g_Designed = real(g_Designed);

pulseSpectrumError = norm(g_Designed-g_design)/max(norm(g_design),eps);
fprintf('Relative squared-spectrum reconstruction error: %.3e\n',pulseSpectrumError);

%% Generate the RRC pulse
[p_RRC,t,filtDelay] = commpyRrcosfilter(L*N,alpha,Tsym,L/Tsym); 
p_RRC = p_RRC/sqrt(sum(abs(p_RRC).^2));
g_RRC = N*(FLN*p_RRC).*(conj(FLN)*conj(p_RRC));
g_RRC = real(g_RRC);

%% Generate the 16-QAM constellation
MOD_TYPE = 'qam'; 
Order = 16;
Constellation = qammod((0:Order-1).',Order,'gray','UnitAveragePower',true);
AvgEnergy = mean(abs(Constellation).^2); 

U = FN'; 
V = eye(N);
tilde_V = V.*conj(V); 

%% Theoretical average squared ACFs with 10,000 coherent integrations, Eq. (36)
TheoAveACF_RRC_M10000 = zeros(L*N,1);
TheoAveACF_Designed_M10000 = zeros(L*N,1);

for k = 0:L*N-1
    gk_RRC = g_RRC(1:N)+(1-g_RRC(1:N))*exp(-1j*2*pi_value*k/L);
    gk_Designed = g_Designed(1:N)+(1-g_Designed(1:N))*exp(-1j*2*pi_value*k/L);
    f_k = exp(-1j*2*pi_value*k*(0:N-1).'/(L*N));
    r1_RRC = abs(gk_RRC.'*conj(f_k))^2;
    r1_Designed = abs(gk_Designed.'*conj(f_k))^2;
    r2_RRC = (kappa-1)/M*(N-2*(1-cos(2*pi_value*k/L))*sum(g_RRC(1:N).*(1-g_RRC(1:N))));
    r2_Designed = (kappa-1)/M*(N-2*(1-cos(2*pi_value*k/L))*sum(g_Designed(1:N).*(1-g_Designed(1:N))));
    TheoAveACF_RRC_M10000(k+1) = r1_RRC+r2_RRC;
    TheoAveACF_Designed_M10000(k+1) = r1_Designed+r2_Designed;
end

normalization_Theo = TheoAveACF_RRC_M10000(1);
TheoAveACF_RRC_M10000 = TheoAveACF_RRC_M10000/normalization_Theo+1e-14;
TheoAveACF_Designed_M10000 = TheoAveACF_Designed_M10000/normalization_Theo+1e-14;
TheoAveACF_RRC_M10000 = fftshift(TheoAveACF_RRC_M10000);
TheoAveACF_Designed_M10000 = fftshift(TheoAveACF_Designed_M10000);

%% Numerical ACFs with 10,000 coherent integrations
if runNumericalSimulation
    Iter = 100;
    SimAveACF_RRC_M10000 = complex(zeros(M,Iter,L*N));
    SimAveACF_Designed_M10000 = complex(zeros(M,Iter,L*N));
    
    for k = 0:L*N-1
        fk = FLN(:,k+1);
        fk_tilde = fk(1:N);
        gk_RRC = g_RRC(1:N)+(1-g_RRC(1:N))*exp(-1j*2*pi_value*k/L);
        gk_Designed = g_Designed(1:N)+(1-g_Designed(1:N))*exp(-1j*2*pi_value*k/L);
    
        for m = 1:M
            for it = 1:Iter
                d = randi([0,Order-1],N,1);
                s = Constellation(d+1);
                VHs = abs(V'*s).^2;
                SimAveACF_RRC_M10000(m,it,k+1) = sum(gk_RRC.*VHs.*conj(fk_tilde));
                SimAveACF_Designed_M10000(m,it,k+1) = sum(gk_Designed.*VHs.*conj(fk_tilde));
            end
        end
    end
    
    %% Coherent averaging
    RkBar_RRC = mean(SimAveACF_RRC_M10000,1);
    RkBar_Designed = mean(SimAveACF_Designed_M10000,1);
    RkBar2_RRC = abs(RkBar_RRC).^2;
    RkBar2_Designed = abs(RkBar_Designed).^2;
    
    Sim_RRC_M10000_avg = squeeze(mean(RkBar2_RRC,2)).';
    Sim_Designed_M10000_avg = squeeze(mean(RkBar2_Designed,2)).';
    
    normalization = Sim_RRC_M10000_avg(1);
    Sim_RRC_M10000_avg = Sim_RRC_M10000_avg/normalization+1e-14;
    Sim_Designed_M10000_avg = Sim_Designed_M10000_avg/normalization+1e-14;
    Sim_RRC_M10000_avg = fftshift(Sim_RRC_M10000_avg);
    Sim_Designed_M10000_avg = fftshift(Sim_Designed_M10000_avg);
end

%% Plot Fig. 6(c)
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
if runNumericalSimulation
    plot(x,10*log10(Sim_RRC_M10000_avg),'--', 'Color','#8A2BE2', 'LineWidth',1.5); hold on;
    plot(x,10*log10(Sim_Designed_M10000_avg),'--', 'Color','#05C349', 'LineWidth',1.5);
end
plot(x,10*log10(TheoAveACF_RRC_M10000),'-', 'Color','#F65314', 'LineWidth',1.5); hold on;
plot(x,10*log10(TheoAveACF_Designed_M10000),'-', 'Color','#00A1F1', 'LineWidth',1.5);

set(gca, 'FontSize',16, 'FontName','Times New Roman');
if runNumericalSimulation
    h_legend = legend('RRC, 10k Coh Integration, Numerical', 'Designed Pulse, 10k Coh Integration, Numerical', 'RRC, 10k Coh Integration, Theoretical', 'Designed Pulse, 10k Coh Integration, Theoretical', 'Interpreter','latex');
else
    h_legend = legend('RRC, 10k Coh Integration, Theoretical', 'Designed Pulse, 10k Coh Integration, Theoretical', 'Interpreter','latex');
end
legendsize = 13;
set(h_legend, 'FontName','Times New Roman', 'FontSize',legendsize, 'FontWeight','normal', 'LineWidth',1, 'Location','southwest');

labelsize = 16;
xlabel('Delay Index', 'FontSize',labelsize, 'FontName','Times New Roman', 'Interpreter','latex');
ylabel('Ambiguity Level (dB)', 'FontSize',labelsize, 'FontName','Times New Roman', 'Interpreter','latex');

xlim([-300,300]);
ylim([-80,0]);
xticks(-300:100:300);
yticks(-80:10:0);

grid on;
set(gca, 'GridLineStyle','--', 'GridAlpha',0.2, 'LineWidth',1, 'GridLineWidth',0.5, 'Layer','bottom');
set(gca, 'Units','normalized');
set(gca, 'Position',[0.125,0.125,0.85,0.86]);

% print(gcf, './Figs/Fig_6c_m.pdf', '-dpdf', '-vector');


function g_opt = solve_iceberg_shaping_psl(N,L,alpha,K_s1)
N_alpha = fix(alpha*N);
N_non_rolloff = N-N_alpha;
N_zeros = floor(N_non_rolloff/2);
N_ones = floor(N_non_rolloff/2);

cvx_begin quiet
    variable g(N) nonnegative
    expressions psl_terms(length(K_s1))

    for indexK = 1:length(K_s1)
        k = K_s1(indexK);
        f_k = exp(-1j*2*pi*k*(0:N-1).'/(L*N));
        gk = g+(1-g)*exp(-1j*2*pi*k/L);
        psl_terms(indexK) = square_abs(f_k'*gk);
    end

    minimize(max(psl_terms))

    subject to
        g(1:N_zeros) == 1;
        g(N-N_ones+1:N) == 0;

        for n = 1:N-1
            g(n+1)-g(n) <= 0;
        end

        sum(g) == N/2;
cvx_end

if strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved')
    fprintf('Optimization successful!\n');
    fprintf('Optimal PSL value: %.12e\n',cvx_optval);
    g_opt = g;
else
    error('CVX failed to solve the PSL problem. Status: %s',cvx_status);
end
end


function mat = FFTmatrix(L)
mat = complex(zeros(L,L));
ll = 0:L-1;
for i = 0:L-1
    mat(i+1,:) = exp(-1j*2*pi*i*ll/L)/sqrt(L);
end
end


function [p,t,filtDelay] = commpyRrcosfilter(N,alpha,Ts,Fs)
T_delta = 1/Fs;
t = ((0:N-1).'-N/2)*T_delta;
p = zeros(N,1);

for x = 1:N
    t_x = t(x);
    if t_x == 0
        p(x) = 1-alpha+4*alpha/pi;
    elseif alpha ~= 0 && t_x == Ts/(4*alpha)
        p(x) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
    elseif alpha ~= 0 && t_x == -Ts/(4*alpha)
        p(x) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
    else
        p(x) = (sin(pi*t_x*(1-alpha)/Ts)+4*alpha*(t_x/Ts)*cos(pi*t_x*(1+alpha)/Ts))/(pi*t_x*(1-(4*alpha*t_x/Ts)^2)/Ts);
    end
end

filtDelay = (N-1)/2;
end
