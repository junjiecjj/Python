%% Fig. 7
% Uncovering the Iceberg in the Sea: Fundamentals of Pulse Shaping and
% Modulation Design for Random ISAC Signals
% Fig. 7 uses the ISL objective in (43), not the PSL objective in (44).

clear;
close all;
clc;
rng(42,'twister');

%% Physical and equivalent parameters
c = 299792458;
Bsys = 122.88e6;
N = 128;
L = 10;
alpha = 0.35;
Order = 16;
deltaF = Bsys/N;
Tu = 1/deltaF;
Tcp = 500e-9;
Tofdm = Tu+Tcp;
Fs = L*Bsys;
Ts = 1/Fs;
deltaRange = c*Ts/2;
Ncp = round(Tcp*Fs);
maximumRangeCP = c*Tcp/2;

Rstrong = 20;
Rweak = 35;
rangeRegion = [26.0993,38.2979];
delayStrong = Rstrong/deltaRange;
delayWeak = Rweak/deltaRange;

% The ISL region is the relative delay from the strong target, not the
% absolute rangeRegion. This gives K_sl=50:150 exactly as in (43).
K_sl = 5*L:15*L;
rangeRegionFromK = Rstrong+[K_sl(1),K_sl(end)]*deltaRange;

SNRdB = -35:5:40;
MonteCarlo = 1000;
profileSNRdB = 20;
strongWeakRatio_dB = 46;% The paper only states a range of 43-46 dB
betaStrong = 1;
betaWeak = 10^(-strongWeakRatio_dB/20);
detectionThreshold_dB = -80;% The exact threshold is not stated in the paper

fprintf('Bandwidth = %.2f MHz\n',Bsys/1e6);
fprintf('Subcarrier spacing = %.2f kHz\n',deltaF/1e3);
fprintf('Useful OFDM duration = %.6f us\n',Tu*1e6);
fprintf('OFDM duration including CP = %.6f us\n',Tofdm*1e6);
fprintf('Range sampling interval = %.6f m\n',deltaRange);
fprintf('CP-limited maximum range = %.6f m\n',maximumRangeCP);
fprintf('Ncp at the oversampled rate = %d samples\n',Ncp);
fprintf('ISL relative-delay region = [%d,%d]\n',K_sl(1),K_sl(end));
fprintf('Corresponding absolute range region = [%.4f,%.4f] m\n',rangeRegionFromK(1),rangeRegionFromK(2));

%% Fourier matrices and modulation bases
FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);
U_SC = eye(N);
U_OFDM = FN';

%% Design the Nyquist pulse by minimizing ISL in (43)
gN = solve_iceberg_shaping_isl(N,L,alpha,K_sl);
g_N = 1-gN;
g_design = [gN;zeros((L-2)*N,1);g_N];
P_spectrum = sqrt(max(real(g_design),0)/N);
p_Designed = FLN'*P_spectrum;
p_Designed = p_Designed/sqrt(sum(abs(p_Designed).^2));

%% Generate the RRC pulse
[p_RRC,t,filtDelay] = commpyRrcosfilter(L*N,alpha,1,L); 
p_RRC = p_RRC/sqrt(sum(abs(p_RRC).^2));

%% Generate the unit-power 16-PSK constellation
Constellation = pskmod((0:Order-1).', Order, 0, 'gray');
AvgEnergy = mean(abs(Constellation).^2); 

%% Monte Carlo ranging simulation for Fig. 7(a)
numSNR = length(SNRdB);
rangeEstimate_RRC_SC = zeros(numSNR, MonteCarlo);
rangeEstimate_Designed_SC = zeros(numSNR, MonteCarlo);
rangeEstimate_RRC_OFDM = zeros(numSNR, MonteCarlo);
rangeEstimate_Designed_OFDM = zeros(numSNR, MonteCarlo);

for indexSNR = 1:numSNR
    fprintf('Monte Carlo simulation: SNR = %d dB\n',SNRdB(indexSNR));

    for it = 1:MonteCarlo
        d = randi([0,Order-1],N,1);
        s = Constellation(d+1);
        x_SC = U_SC*s;
        x_OFDM = U_OFDM*s;

        tx_RRC_SC = pulseShaping(x_SC,p_RRC,L);
        tx_Designed_SC = pulseShaping(x_SC,p_Designed,L);
        tx_RRC_OFDM = pulseShaping(x_OFDM,p_RRC,L);
        tx_Designed_OFDM = pulseShaping(x_OFDM,p_Designed,L);

        tx_RRC_SC = tx_RRC_SC/sqrt(sum(abs(tx_RRC_SC).^2)/N);
        tx_Designed_SC = tx_Designed_SC/sqrt(sum(abs(tx_Designed_SC).^2)/N);
        tx_RRC_OFDM = tx_RRC_OFDM/sqrt(sum(abs(tx_RRC_OFDM).^2)/N);
        tx_Designed_OFDM = tx_Designed_OFDM/sqrt(sum(abs(tx_Designed_OFDM).^2)/N);

        rangeEstimate_RRC_SC(indexSNR,it) = simulateRanging(tx_RRC_SC,delayStrong,delayWeak,betaStrong,betaWeak,SNRdB(indexSNR),deltaRange,rangeRegion,detectionThreshold_dB);
        rangeEstimate_Designed_SC(indexSNR,it) = simulateRanging(tx_Designed_SC,delayStrong,delayWeak,betaStrong,betaWeak,SNRdB(indexSNR),deltaRange,rangeRegion,detectionThreshold_dB);
        rangeEstimate_RRC_OFDM(indexSNR,it) = simulateRanging(tx_RRC_OFDM,delayStrong,delayWeak,betaStrong,betaWeak,SNRdB(indexSNR),deltaRange,rangeRegion,detectionThreshold_dB);
        rangeEstimate_Designed_OFDM(indexSNR,it) = simulateRanging(tx_Designed_OFDM,delayStrong,delayWeak,betaStrong,betaWeak,SNRdB(indexSNR),deltaRange,rangeRegion,detectionThreshold_dB);
    end
end

RMSE_RRC_SC = sqrt(mean((rangeEstimate_RRC_SC-Rweak).^2,2));
RMSE_Designed_SC = sqrt(mean((rangeEstimate_Designed_SC-Rweak).^2,2));
RMSE_RRC_OFDM = sqrt(mean((rangeEstimate_RRC_OFDM-Rweak).^2,2));
RMSE_Designed_OFDM = sqrt(mean((rangeEstimate_Designed_OFDM-Rweak).^2,2));

%% Generate representative range profiles for Fig. 7(b)
d = randi([0,Order-1],N,1);
s = Constellation(d+1);
x_SC = U_SC*s;
x_OFDM = U_OFDM*s;
tx_RRC_SC = pulseShaping(x_SC,p_RRC,L); 
tx_RRC_SC = tx_RRC_SC/sqrt(sum(abs(tx_RRC_SC).^2)/N);
tx_Designed_SC = pulseShaping(x_SC,p_Designed,L); 
tx_Designed_SC = tx_Designed_SC/sqrt(sum(abs(tx_Designed_SC).^2)/N);
tx_RRC_OFDM = pulseShaping(x_OFDM,p_RRC,L); 
tx_RRC_OFDM = tx_RRC_OFDM/sqrt(sum(abs(tx_RRC_OFDM).^2)/N);
tx_Designed_OFDM = pulseShaping(x_OFDM,p_Designed,L); 
tx_Designed_OFDM = tx_Designed_OFDM/sqrt(sum(abs(tx_Designed_OFDM).^2)/N);

[~,rangeProfile_RRC_SC] = simulateRanging(tx_RRC_SC,delayStrong,delayWeak,betaStrong,betaWeak,profileSNRdB,deltaRange,rangeRegion,detectionThreshold_dB);
[~,rangeProfile_Designed_SC] = simulateRanging(tx_Designed_SC,delayStrong,delayWeak,betaStrong,betaWeak,profileSNRdB,deltaRange,rangeRegion,detectionThreshold_dB);
[~,rangeProfile_RRC_OFDM] = simulateRanging(tx_RRC_OFDM,delayStrong,delayWeak,betaStrong,betaWeak,profileSNRdB,deltaRange,rangeRegion,detectionThreshold_dB);
[~,rangeProfile_Designed_OFDM] = simulateRanging(tx_Designed_OFDM,delayStrong,delayWeak,betaStrong,betaWeak,profileSNRdB,deltaRange,rangeRegion,detectionThreshold_dB);

rangeAxis = (0:L*N-1)*deltaRange;

%% Plot Fig. 7(a)
width = 8;%设置图宽
height = 6;%设置图高
fontsize = 14;%设置图中字体大小
linewidth = 2;%设置线宽
markersize = 10;%标记大小
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');

figure(1);
set(gcf, 'Units','inches');
set(gcf, 'Color','white');
set(gcf, 'Renderer','painters');
set(gcf, 'PaperUnits','inches');
set(gcf, 'PaperPosition',[0,0,width,height]);
set(gcf, 'PaperSize',[width,height]);

semilogy(SNRdB,RMSE_RRC_SC,'-.', 'Color','#00A1F1', 'LineWidth',1.5); hold on;
semilogy(SNRdB,RMSE_Designed_SC,'-.', 'Color','#8A2BE2', 'LineWidth',1.5);
semilogy(SNRdB,RMSE_RRC_OFDM,'>-', 'Color','#00A1F1', 'LineWidth',1.5, 'MarkerSize',8, 'MarkerFaceColor','white');
semilogy(SNRdB,RMSE_Designed_OFDM,'o-', 'Color','#8A2BE2', 'LineWidth',1.5, 'MarkerSize',7, 'MarkerFaceColor','white');

set(gca, 'FontSize',16, 'FontName','Times New Roman');
h_legend = legend('RRC, SC', 'Designed Pulse, SC', 'RRC, OFDM', 'Designed Pulse, OFDM', 'Interpreter','latex');
set(h_legend, 'FontName','Times New Roman', 'FontSize',13, 'FontWeight','normal', 'LineWidth',1, 'Location','southwest');
xlabel('SNR (dB)', 'FontSize',16, 'FontName','Times New Roman', 'Interpreter','latex');
ylabel('RMSE (m)', 'FontSize',16, 'FontName','Times New Roman', 'Interpreter','latex');
xlim([-35,25]);
ylim([1e-3,1e1]);
xticks(-30:10:25);
yticks([1e-3,1e-2,1e-1,1e0,1e1]);
grid on;
set(gca, 'GridLineStyle','--', 'GridAlpha',0.2, 'LineWidth',1, 'GridLineWidth',0.5, 'Layer','bottom');
set(gca, 'Units','normalized');
set(gca, 'Position',[0.125,0.125,0.85,0.86]);
% print(gcf, './Figs/Fig_7a_m.pdf', '-dpdf', '-vector');

%% Plot Fig. 7(b)
figure(2);
set(gcf, 'Units','inches');
set(gcf, 'Color','white');
set(gcf, 'Renderer','painters');
set(gcf, 'PaperUnits','inches');
set(gcf, 'PaperPosition',[0,0,width,8]);
set(gcf, 'PaperSize',[width,8]);

subplot(2,1,1);
plotRangeRegion(rangeRegion,-70,0); hold on;
plot(rangeAxis,rangeProfile_RRC_SC,'--', 'Color','#00A1F1', 'LineWidth',1.5);
plot(rangeAxis,rangeProfile_Designed_SC,'-', 'Color','#F65314', 'LineWidth',1.5);
xline(Rstrong,'-.', 'Color','#404040', 'LineWidth',1.5);
xline(Rweak,'-.', 'Color','#404040', 'LineWidth',1.5);
set(gca, 'FontSize',16, 'FontName','Times New Roman');
h_legend = legend('Delay Region of Interest', 'RRC, SC', 'Designed Pulse, SC', 'Ground-Truth', 'Interpreter','latex');
set(h_legend, 'FontName','Times New Roman', 'FontSize',10, 'FontWeight','normal', 'LineWidth',1, 'Location','northeast');
ylabel('Amplitude (dB)', 'FontSize',16, 'FontName','Times New Roman', 'Interpreter','latex');
xlim([10,75]); ylim([-70,0]); xticks(10:10:70); yticks(-60:20:0); grid on;
set(gca, 'GridLineStyle','--', 'GridAlpha',0.2, 'LineWidth',1, 'GridLineWidth',0.5, 'Layer','bottom');

subplot(2,1,2);
plotRangeRegion(rangeRegion,-70,0); hold on;
plot(rangeAxis,rangeProfile_RRC_OFDM,'--', 'Color','#00A1F1', 'LineWidth',1.5);
plot(rangeAxis,rangeProfile_Designed_OFDM,'-', 'Color','#F65314', 'LineWidth',1.5);
xline(Rstrong,'-.', 'Color','#404040', 'LineWidth',1.5);
xline(Rweak,'-.', 'Color','#404040', 'LineWidth',1.5);
set(gca, 'FontSize',16, 'FontName','Times New Roman');
h_legend = legend('Delay Region of Interest', 'RRC, OFDM', 'Designed Pulse, OFDM', 'Ground-Truth', 'Interpreter','latex');
set(h_legend, 'FontName','Times New Roman', 'FontSize',10, 'FontWeight','normal', 'LineWidth',1, 'Location','northeast');
xlabel('Range (m)', 'FontSize',16, 'FontName','Times New Roman', 'Interpreter','latex');
ylabel('Amplitude (dB)', 'FontSize',16, 'FontName','Times New Roman', 'Interpreter','latex');
xlim([10,75]); ylim([-70,0]); xticks(10:10:70); yticks(-60:20:0); grid on;
set(gca, 'GridLineStyle','--', 'GridAlpha',0.2, 'LineWidth',1, 'GridLineWidth',0.5, 'Layer','bottom');
% print(gcf, './Figs/Fig_7b_m.pdf', '-dpdf', '-vector');




function g_opt = solve_iceberg_shaping_isl(N,L,alpha,K_sl)
    N_alpha = fix(alpha*N);
    N_non_rolloff = N-N_alpha;
    N_zeros = floor(N_non_rolloff/2);
    N_ones = floor(N_non_rolloff/2);
    
    cvx_begin quiet
        variable g(N) nonnegative
        expressions isl_terms(length(K_sl))
    
        for indexK = 1:length(K_sl)
            k = K_sl(indexK);
            f_k = exp(-1j*2*pi*k*(0:N-1).'/(L*N));
            gk = g+(1-g)*exp(-1j*2*pi*k/L);
            isl_terms(indexK) = square_abs(f_k'*gk);
        end
    
        minimize(sum(isl_terms))
    
        subject to
            g(1:N_zeros) == 1;
            g(N-N_ones+1:N) == 0;
    
            for n = 1:N-1
                g(n+1)-g(n) <= 0;
            end
    
            sum(g) == N/2;
    cvx_end
    
    if strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved')
        fprintf('ISL optimization successful!\n');
        fprintf('Optimal ISL value: %.12e\n',cvx_optval);
        g_opt = g;
    else
        error('CVX failed to solve the ISL problem. Status: %s',cvx_status);
    end
end


function tx = pulseShaping(x,p,L)
    N = length(x);
    xup = complex(zeros(L*N,1));
    xup(1:L:end) = x;
    tx = ifft(fft(xup).*fft(p));
end


function [rangeEstimate,rangeProfile_dB] = simulateRanging(tx,delayStrong,delayWeak,betaStrong,betaWeak,SNRdB,deltaRange,rangeRegion,detectionThreshold_dB)
    K = length(tx);
    P = mean(abs(tx).^2);
    sigma2 = P*abs(betaWeak)^2/10^(SNRdB/10);
    echo = betaStrong*fractionalCircularDelay(tx,delayStrong)+betaWeak*fractionalCircularDelay(tx,delayWeak);
    noise = sqrt(sigma2/2)*(randn(K,1)+1j*randn(K,1));
    y = echo+noise;
    rangeProfile = abs(ifft(fft(y).*conj(fft(tx)))).^2;
    rangeProfile_dB = 10*log10(rangeProfile/max(rangeProfile)+1e-14);
    rangeEstimate = estimateWeakTargetRange(rangeProfile,deltaRange,rangeRegion,detectionThreshold_dB);
    rangeProfile_dB = rangeProfile_dB.';
end

function y = fractionalCircularDelay(x,delay)
    K = length(x);
    frequencyIndex = ifftshift((-floor(K/2):ceil(K/2)-1).');
    y = ifft(fft(x).*exp(-1j*2*pi*frequencyIndex*delay/K));
end

function rangeEstimate = estimateWeakTargetRange(rangeProfile,deltaRange,rangeRegion,detectionThreshold_dB)
    K = length(rangeProfile);
    rangeAxis = (0:K-1).'*deltaRange;
    searchIndex = find(rangeAxis >= rangeRegion(1) & rangeAxis <= rangeRegion(2));
    threshold = max(rangeProfile)*10^(detectionThreshold_dB/10);
    localPeakIndex = searchIndex(rangeProfile(searchIndex) >= threshold & rangeProfile(searchIndex) >= rangeProfile(mod(searchIndex-2,K)+1) & rangeProfile(searchIndex) >= rangeProfile(mod(searchIndex,K)+1));
    
    if isempty(localPeakIndex)
        [~,relativeIndex] = max(rangeProfile(searchIndex));
        peakIndex = searchIndex(relativeIndex);
    else
        [~,relativeIndex] = max(rangeProfile(localPeakIndex));
        peakIndex = localPeakIndex(relativeIndex);
    end
    previousIndex = mod(peakIndex-2,K)+1;
    nextIndex = mod(peakIndex,K)+1;
    yPrevious = log(max(rangeProfile(previousIndex),realmin));
    yPeak = log(max(rangeProfile(peakIndex),realmin));
    yNext = log(max(rangeProfile(nextIndex),realmin));
    denominator = yPrevious-2*yPeak+yNext;
    if abs(denominator) > eps
        fractionalOffset = 0.5*(yPrevious-yNext)/denominator;
        fractionalOffset = max(min(fractionalOffset,1),-1);
    else
        fractionalOffset = 0;
    end
    rangeEstimate = ((peakIndex-1)+fractionalOffset)*deltaRange;
end

function plotRangeRegion(rangeRegion,yMinimum,yMaximum)
    patch([rangeRegion(1),rangeRegion(2),rangeRegion(2),rangeRegion(1)],[yMinimum,yMinimum,yMaximum,yMaximum],[0.90,0.88,1.00], 'EdgeColor',[0.65,0.55,0.90], 'FaceAlpha',0.45);
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