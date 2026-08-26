%% Fig. 6(a)
% Uncovering the Iceberg in the Sea: Fundamentals of Pulse Shaping and
% Modulation Design for Random ISAC Signals
%
% OFDM, 16-QAM, N = 128, L = 10, alpha = 0.35.
% The pulse is first designed through the PSL problem in (44)--(49), and
% then used in the theoretical and numerical ACF calculations inherited
% from Fig2.py.
%
% CVX is required: http://cvxr.com/cvx/

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

%% Parameters in Fig. 6(a)
Tsym = 1;
pi_value = pi;
N = 128;
L = 10;
alpha = 0.35;
kappa = 1.32;

FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);

%% Generate the optimized pulse according to Fig. 6(d)
% The delay region [5,15] is expressed in symbol-delay units, whereas the
% discrete ACF is sampled at Ts = T/L.
K_s1 = 5*L:15*L;
gN = solve_iceberg_shaping_psl(N,L,alpha,K_s1);

g_N = 1-gN;
g_design = [gN;zeros((L-2)*N,1);g_N];

% g_design = N*|F_{LN}p|^2. A zero spectral phase is selected to recover
% one valid pulse having exactly the optimized squared spectrum.
P_spectrum = sqrt(max(real(g_design),0)/N);
p = FLN'*P_spectrum;
p = p/sqrt(sum(abs(p).^2));
 
g = N*(FLN*p).*(conj(FLN)*conj(p));
g = real(g);

pulseSpectrumError = norm(g-g_design)/max(norm(g_design),eps);
fprintf('Relative squared-spectrum reconstruction error: %.3e\n', pulseSpectrumError);

%% OFDM theoretical average squared ACF, Eq. (36)
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

% Fig. 6(a) displays numerical realizations. Keep the full 3-D coherent-
% integration array, with one outer realization of the 10,000-slot result.
Iter = 1000;

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

for k = 0:L*N-1
    fk = FLN(:,k+1);
    fk_tilde = fk(1:N);
    gk = g(1:N)+(1-g(1:N))*exp(-1j*2*pi_value*k/L);

    for m = 1:M
        for it = 1:Iter
            d = randi([0,Order-1],N,1);
            s = Constellation(d+1);
            VHs = abs(V'*s).^2;

            % Do not square here. Average the complex ACF over M first.
            SimAveACF_OFDM_M10000(m,it,k+1) = sum(gk.*VHs.*conj(fk_tilde));
        end
    end
end

%% Coherent averaging, Eq. (33)
RkBar = mean(SimAveACF_OFDM_M10000,1);
RkBar2 = abs(RkBar).^2;

Sim_M10000_avg = squeeze(mean(RkBar2,2)).';
Sim_M10000_avg = Sim_M10000_avg/max(Sim_M10000_avg)+1e-14;
Sim_M10000_avg = fftshift(Sim_M10000_avg);

%% Plot Fig. 6(a)
x = -N*L/2:N*L/2-1;

figure('Color','w','Position',[100,100,760,620]);
hold on;

plot(x,10*log10(Sim_M1_avg),'-', 'Color',[0,0.4470,0.7410],'LineWidth',1.2, 'DisplayName','No Integration, Numerical');
plot(x,10*log10(TheoAveACF_OFDM_M1),':', 'Color',[0,0.4470,0.7410],'LineWidth',1.8, 'DisplayName','No Integration, Theoretical');

plot(x,10*log10(Sim_M10000_avg),'-', 'Color',[0.8500,0.3250,0.0980],'LineWidth',1.2, 'DisplayName','10k Coh Integration, Numerical');
plot(x,10*log10(TheoAveACF_OFDM_M10000),':', 'Color',[0.8500,0.3250,0.0980],'LineWidth',1.8, 'DisplayName','10k Coh Integration, Theoretical');

plot(x,10*log10(TheoAveACF_Iceberg),'k--','LineWidth',1.2, 'DisplayName','"Iceberg" of the Designed Pulse');

xlabel('Delay Index');
ylabel('Ambiguity Level (dB)');
xlim([-300,300]);
ylim([-120,0]);
xticks(-300:50:300);
yticks(-120:20:0);
grid on;
box on;
legend('Location','south','EdgeColor','black');
set(gca,'FontName','Times New Roman','FontSize',14);

exportgraphics(gcf,'Fig6_a.png','Resolution',300);
exportgraphics(gcf,'Fig6_a.pdf','ContentType','vector');

%% ===========================================
width = 8;%设置图宽，这个不用改
height = 4;%设置图高，这个不用改
fontsize = 14;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
markersize = 10;%标记的大小，按照个人喜好设置。
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
% ===========================================
figure(1);
% gca表示对axes的设置；  gcf表示对figure的设置
set(gcf, 'Units', 'inches');
% set(gcf, 'Position', [0, 0, width, height]);
set(gcf, 'Color', 'white'); % 设置背景是白色的 原先是灰色的 论文里面不好看
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);

plot(x,10*log10(Sim_M1_avg),'-', 'Color','#F65314', 'LineWidth', 2); hold on;
plot(x,10*log10(TheoAveACF_OFDM_M1),':', 'Color', '#F65314', 'LineWidth', 2);
plot(x,10*log10(Sim_M10000_avg),'-', 'Color','#00A1F1', 'LineWidth', 2);
plot(x,10*log10(TheoAveACF_OFDM_M10000),':', 'Color', '#00A1F1', 'LineWidth', 1.5);
plot(x,10*log10(TheoAveACF_Iceberg),'k--', 'Color', '#B0B0B0', 'LineWidth', 1.5);


% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',16,'FontName','Times New Roman');
h_legend =  legend('No Integration, Numerical', 'No Integration, Theoretical', '10k Coh Integration, Numerical', '10k Coh Integration, Theoretical', '"Iceberg" of the Designed Pulse','Interpreter', 'latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
% set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 16;
xlabel('Delay Index', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('Ambiguity Level (dB)', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');

xlim([-300,300]);
ylim([-120,0]);
xticks(-300:50:300);
yticks(-120:20:0);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

% print(gcf, 'Fig6_a.pdf', '-dpdf', '-vector');






%%  
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
        % Reversed FFT-bin ordering used by the numerical implementation:
        % 1 -> monotonically decreasing roll-off -> 0.
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
for i = 0:L-1
    for j = 0:L-1
        mat(i+1,j+1) = exp(-1j*2*pi*i*j/L)/sqrt(L);
    end
end
end