%% Fig. 7 reproduction -- integer-delay, ISL-designed pulse
% Uncovering the Iceberg in the Sea: Fundamentals of Pulse Shaping and Modulation Design for Random ISAC Signals
% CVX is required. All target delays and range estimates are on the sampling grid.
clear;
clc;
close all;
rng(42);

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
Rstrong = 20;
Rweak = 35;
rangeRegion = [26.0993,38.2979];
delayStrong = round(Rstrong/deltaRange);
delayWeak = round(Rweak/deltaRange);
RstrongGrid = delayStrong*deltaRange;
RweakGrid = delayWeak*deltaRange;
relativeRangeRegion = rangeRegion-RstrongGrid;
KslBounds = round(relativeRangeRegion/deltaRange);
K_sl = KslBounds(1):KslBounds(2);
delayROI = delayStrong+K_sl;
rangeROIGrid = delayROI*deltaRange;
fprintf('Fs = %.3f MHz, range-bin spacing = %.6f m, Ncp = %d samples\n',Fs/1e6,deltaRange,Ncp);
fprintf('Strong target: delay %d, grid range %.6f m\n',delayStrong,RstrongGrid);
fprintf('Weak target: delay %d, grid range %.6f m\n',delayWeak,RweakGrid);
fprintf('Relative ISL region K_sl = [%d,%d], absolute grid ROI = [%.6f,%.6f] m\n',K_sl(1),K_sl(end),rangeROIGrid(1),rangeROIGrid(end));

%% Simulation parameters
SNRdB = -10:5:30;
MonteCarlo = 10000;
profileSNRdB = 20;
strongWeakRatio_dB = 46;
betaStrongMagnitude = 1;
betaWeakMagnitude = 10^(-strongWeakRatio_dB/20);
Ptx = 1;
randomTargetPhase = true;
detectionThreshold_dB = -80;

%% Fourier matrices, RRC pulse and ISL-designed pulse
FLN = FFTmatrix(L*N);
FN = FFTmatrix(N);
[p_RRC,~,~] = commpyRrcosfilter(L*N,alpha,1,L);
p_RRC = p_RRC/sqrt(sum(abs(p_RRC).^2));
gN = solve_iceberg_shaping_isl(N,L,alpha,K_sl);
g_design = [gN; zeros((L-2)*N,1); 1-gN];
p_Designed = FLN'*sqrt(max(real(g_design),0)/N);
p_Designed = p_Designed/sqrt(sum(abs(p_Designed).^2));
fprintf('RRC pulse energy = %.12f, designed pulse energy = %.12f\n',sum(abs(p_RRC).^2),sum(abs(p_Designed).^2));

%% 16-PSK modulation and SC/OFDM modulation matrices
Constellation = pskmod((0:Order-1).',Order,pi/Order,'gray');
Constellation = Constellation/sqrt(mean(abs(Constellation).^2));
U_SC = eye(N);
U_OFDM = FN';

%% Monte Carlo range estimation
numSNR = length(SNRdB);
RangeEstimate_RRC_SC = zeros(numSNR,MonteCarlo);
RangeEstimate_Designed_SC = zeros(numSNR,MonteCarlo);
RangeEstimate_RRC_OFDM = zeros(numSNR,MonteCarlo);
RangeEstimate_Designed_OFDM = zeros(numSNR,MonteCarlo);
for indexSNR = 1:numSNR
    noiseVariance = Ptx*betaWeakMagnitude^2/10^(SNRdB(indexSNR)/10);
    for indexMC = 1:MonteCarlo
        symbolIndex = randi([0,Order-1],N,1);
        s = Constellation(symbolIndex+1);
        if randomTargetPhase
            betaStrong = betaStrongMagnitude*exp(1j*2*pi*rand);
            betaWeak = betaWeakMagnitude*exp(1j*2*pi*rand);
        else
            betaStrong = betaStrongMagnitude;
            betaWeak = betaWeakMagnitude;
        end
        tx_RRC_SC = formTransmitWaveform(U_SC*s,p_RRC,L,Ptx);
        tx_Designed_SC = formTransmitWaveform(U_SC*s,p_Designed,L,Ptx);
        tx_RRC_OFDM = formTransmitWaveform(U_OFDM*s,p_RRC,L,Ptx);
        tx_Designed_OFDM = formTransmitWaveform(U_OFDM*s,p_Designed,L,Ptx);
        RangeEstimate_RRC_SC(indexSNR,indexMC) = oneRangeEstimate(tx_RRC_SC,betaStrong,betaWeak,delayStrong,delayWeak,noiseVariance,delayROI,deltaRange,detectionThreshold_dB);
        RangeEstimate_Designed_SC(indexSNR,indexMC) = oneRangeEstimate(tx_Designed_SC,betaStrong,betaWeak,delayStrong,delayWeak,noiseVariance,delayROI,deltaRange,detectionThreshold_dB);
        RangeEstimate_RRC_OFDM(indexSNR,indexMC) = oneRangeEstimate(tx_RRC_OFDM,betaStrong,betaWeak,delayStrong,delayWeak,noiseVariance,delayROI,deltaRange,detectionThreshold_dB);
        RangeEstimate_Designed_OFDM(indexSNR,indexMC) = oneRangeEstimate(tx_Designed_OFDM,betaStrong,betaWeak,delayStrong,delayWeak,noiseVariance,delayROI,deltaRange,detectionThreshold_dB);
    end
    fprintf('SNR = %3d dB completed\n',SNRdB(indexSNR));
end
RMSE_RRC_SC = sqrt(mean((RangeEstimate_RRC_SC-RweakGrid).^2,2));
RMSE_Designed_SC = sqrt(mean((RangeEstimate_Designed_SC-RweakGrid).^2,2));
RMSE_RRC_OFDM = sqrt(mean((RangeEstimate_RRC_OFDM-RweakGrid).^2,2));
RMSE_Designed_OFDM = sqrt(mean((RangeEstimate_Designed_OFDM-RweakGrid).^2,2));

%% One representative realization for Fig. 7(b)

symbolIndex = randi([0,Order-1],N,1);
s = Constellation(symbolIndex+1);
betaStrong = betaStrongMagnitude;
betaWeak = betaWeakMagnitude;
profileNoiseVariance = Ptx*betaWeakMagnitude^2/10^(profileSNRdB/10);
tx_RRC_SC = formTransmitWaveform(U_SC*s,p_RRC,L,Ptx);
tx_Designed_SC = formTransmitWaveform(U_SC*s,p_Designed,L,Ptx);
tx_RRC_OFDM = formTransmitWaveform(U_OFDM*s,p_RRC,L,Ptx);
tx_Designed_OFDM = formTransmitWaveform(U_OFDM*s,p_Designed,L,Ptx);
Profile_RRC_SC = oneRangeProfile(tx_RRC_SC,betaStrong,betaWeak,delayStrong,delayWeak,profileNoiseVariance);
Profile_Designed_SC = oneRangeProfile(tx_Designed_SC,betaStrong,betaWeak,delayStrong,delayWeak,profileNoiseVariance);
Profile_RRC_OFDM = oneRangeProfile(tx_RRC_OFDM,betaStrong,betaWeak,delayStrong,delayWeak,profileNoiseVariance);
Profile_Designed_OFDM = oneRangeProfile(tx_Designed_OFDM,betaStrong,betaWeak,delayStrong,delayWeak,profileNoiseVariance);
rangeAxis = (0:L*N-1)*deltaRange;

%% Plot Fig. 7(a)
width = 8;
height = 6;
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
semilogy(SNRdB,RMSE_RRC_SC,'-.','Color','#36A2B4','LineWidth',linewidth);
hold on;
semilogy(SNRdB,RMSE_Designed_SC,'-.','Color','#5A50C8','LineWidth',linewidth);
semilogy(SNRdB,RMSE_RRC_OFDM,'>-','Color','#36A2B4','LineWidth',linewidth,'MarkerSize',markersize,'MarkerFaceColor','white');
semilogy(SNRdB,RMSE_Designed_OFDM,'o-','Color','#5A50C8','LineWidth',linewidth,'MarkerSize',markersize,'MarkerFaceColor','white');
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend('RRC, SC','Designed Pulse, SC','RRC, OFDM','Designed Pulse, OFDM');
set(h_legend,'FontName','Times New Roman','FontSize',13,'FontWeight','normal','LineWidth',1,'Location','southwest');
xlabel('SNR (dB)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
ylabel('RMSE (m)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([-10,30]);
ylim([1e-3,1e1]);
xticks(-10:10:30);
grid on;
set(gca,'GridLineStyle','--','Gridalpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.125,0.125,0.85,0.86]);

%% Plot Fig. 7(b)
figure(2);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

subplot(2,1,1);
fill([rangeRegion(1),rangeRegion(2),rangeRegion(2),rangeRegion(1)],[-70,-70,0,0],[0.95,0.94,1],'EdgeColor',[0.65,0.55,1],'LineWidth',0.5,'DisplayName','Delay Region of Interest');
hold on;
plot(rangeAxis,10*log10(Profile_RRC_SC+1e-14),'--','Color','#00A1F1','LineWidth',1.5,'DisplayName','RRC, SC');
plot(rangeAxis,10*log10(Profile_Designed_SC+1e-14),'-','Color','#F65314','LineWidth',1.5,'DisplayName','Designed Pulse, SC');
xline(RstrongGrid,'--','Color','#808080','LineWidth',1.5,'DisplayName','Ground-Truth');
xline(RweakGrid,'--','Color','#808080','LineWidth',1.5,'HandleVisibility','off');
set(gca,'FontSize',fontsize,'FontName','Times New Roman');
ylabel('Amplitude (dB)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([10,75]);
ylim([-70,0]);
grid on;
set(gca,'GridLineStyle','--','Gridalpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
h_legend = legend('Location','southeast');
set(h_legend,'FontName','Times New Roman','FontSize',10,'FontWeight','normal','LineWidth',1);

subplot(2,1,2);
fill([rangeRegion(1),rangeRegion(2),rangeRegion(2),rangeRegion(1)],[-70,-70,0,0],[0.95,0.94,1],'EdgeColor',[0.65,0.55,1],'LineWidth',0.5,'DisplayName','Delay Region of Interest');
hold on;
plot(rangeAxis,10*log10(Profile_RRC_OFDM+1e-14),'--','Color','#00A1F1','LineWidth',1.5,'DisplayName','RRC, OFDM');
plot(rangeAxis,10*log10(Profile_Designed_OFDM+1e-14),'-','Color','#F65314','LineWidth',1.5,'DisplayName','Designed Pulse, OFDM');
xline(RstrongGrid,'--','Color','#808080','LineWidth',1.5,'DisplayName','Ground-Truth');
xline(RweakGrid,'--','Color','#808080','LineWidth',1.5,'HandleVisibility','off');
set(gca,'FontSize',fontsize,'FontName','Times New Roman');
xlabel('Range (m)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
ylabel('Amplitude (dB)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([10,75]);
ylim([-70,0]);
grid on;
set(gca,'GridLineStyle','--','Gridalpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
h_legend = legend('Location','northeast');
set(h_legend,'FontName','Times New Roman','FontSize',10,'FontWeight','normal','LineWidth',1);

%% Local functions
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
    if ~(strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved'))
        error('CVX failed: %s',cvx_status);
    end
    fprintf('ISL optimization status: %s, objective = %.12e\n',cvx_status,cvx_optval);
    g_opt = g;
end

function tx = formTransmitWaveform(x,p,L,Ptx)
    N = length(x);
    xUp = complex(zeros(L*N,1));
    xUp(1:L:end) = x;
    tx = ifft(fft(xUp).*fft(p));
    tx = sqrt(Ptx)*tx/sqrt(sum(abs(tx).^2));
end

function rangeEstimate = oneRangeEstimate1(tx,betaStrong,betaWeak,delayStrong,delayWeak,noiseVariance,delayROI,deltaRange,detectionThreshold_dB)
    noise = sqrt(noiseVariance/2)*(randn(size(tx))+1j*randn(size(tx)));
    y = betaStrong*circshift(tx,delayStrong)+betaWeak*circshift(tx,delayWeak)+noise;
    profile = abs(ifft(fft(y).*conj(fft(tx)))).^2;
    searchProfile = profile(delayROI+1);
    threshold = max(searchProfile)*10^(detectionThreshold_dB/10);
    leftProfile = [searchProfile(1); searchProfile(1:end-1)];
    rightProfile = [searchProfile(2:end); searchProfile(end)];
    candidate = find(searchProfile>=leftProfile & searchProfile>=rightProfile & searchProfile>=threshold);
    if isempty(candidate)
        [~,candidate] = max(searchProfile);
    else
        [~,largestCandidate] = max(searchProfile(candidate));
        candidate = candidate(largestCandidate);
    end
    delayEstimate = delayROI(candidate);
    rangeEstimate = delayEstimate*deltaRange;
end

function rangeEstimate = oneRangeEstimate(tx,betaStrong,betaWeak,delayStrong,delayWeak,noiseVariance,delayROI,deltaRange,detectionThreshold_dB)

    noise = sqrt(noiseVariance/2)*(randn(size(tx))+1j*randn(size(tx)));
    y = betaStrong*circshift(tx,delayStrong) + betaWeak*circshift(tx,delayWeak) + noise;
    rangeProfile = abs(ifft(fft(y).*conj(fft(tx)))).^2;
    searchProfile = rangeProfile(delayROI+1);
    [~,peakIndex] = max(searchProfile);
    delayEstimate = delayROI(peakIndex);
    rangeEstimate = delayEstimate*deltaRange;

end

function profile = oneRangeProfile(tx,betaStrong,betaWeak,delayStrong,delayWeak,noiseVariance,delayROI,deltaRange)
    noise = sqrt(noiseVariance/2)*(randn(size(tx))+1j*randn(size(tx)));
    y = betaStrong*circshift(tx,delayStrong)+betaWeak*circshift(tx,delayWeak)+noise;
    profile = abs(ifft(fft(y).*conj(fft(tx)))).^2;
    profile = profile/max(profile);
end

function mat = FFTmatrix(K)
    mat = complex(zeros(K,K));
    kk = 0:K-1;
    for index = 0:K-1
        mat(index+1,:) = exp(-1j*2*pi*index*kk/K)/sqrt(K);
    end
end

function [p,t,filtDelay] = commpyRrcosfilter(K,alpha,Tsym,FsPulse)
    T_delta = 1/FsPulse;
    t = ((0:K-1).'-K/2)*T_delta;
    p = zeros(K,1);
    tolerance = 1e-12;
    for index = 1:K
        t_x = t(index);
        if abs(t_x)<tolerance
            p(index) = 1-alpha+4*alpha/pi;
        elseif alpha~=0 && abs(abs(t_x)-Tsym/(4*alpha))<tolerance
            p(index) = alpha/sqrt(2)*((1+2/pi)*sin(pi/(4*alpha))+(1-2/pi)*cos(pi/(4*alpha)));
        else
            p(index) = (sin(pi*t_x*(1-alpha)/Tsym)+4*alpha*(t_x/Tsym)*cos(pi*t_x*(1+alpha)/Tsym))/(pi*t_x*(1-(4*alpha*t_x/Tsym)^2)/Tsym);
        end
    end
    filtDelay = (K-1)/2;
end
