%% Fig. 7 reproduction without oversampling
% Sensing With Communication Signals: From Information Theory to Signal Processing
% L = 1: the sampled Nyquist model reduces to xTilde = x = U*s.
% Transmitter: s -> OFDM modulation -> add CP.
% Sensing receiver: remove CP -> periodic waveform matched filtering.

clear;
clc;
close all;
rng(42);

%% Parameters used for the three-target example
Order = 16;
SNRdB = 0;
targetRange = [10,20,25];
targetAmplitude_dB = [0,-10,-30];
targetPhase = [0,0,0];
Q = length(targetRange);
Iter = 1000;

%% Discrete waveform parameters without oversampling
N = 4096;
L = 1;
rangeSampleSpacing = 0.5;
desiredCPRange = 40;
c = 299792458;

%% Target amplitudes
gamma = 10.^(targetAmplitude_dB/20).*exp(1j*targetPhase);

%% Physical distance, propagation delay and discrete delay index
Fs = c/(2*rangeSampleSpacing);
Ts = 1/Fs;
targetPropagationDelay = 2*targetRange/c;
targetDelaySample = round(targetPropagationDelay/Ts);
targetRangeGrid = targetDelaySample*c*Ts/2;
targetRangeError = targetRangeGrid-targetRange;
maxTargetDelaySample = max(targetDelaySample);

%% CP length is defined at the same rate as x because L = 1
NcpFromDesiredRange = ceil(desiredCPRange/rangeSampleSpacing);
Ncp = max(NcpFromDesiredRange,maxTargetDelaySample+1);
Tcp = Ncp*Ts;
cpSupportedRange = Ncp*c*Ts/2;
lengthBlock = N;

fprintf('Sampling rate Fs = %.6f MHz\n',Fs/1e6);
fprintf('Sampling interval Ts = %.6f ns\n',Ts*1e9);
fprintf('Range grid = %.6f m/sample\n',rangeSampleSpacing);
fprintf('Ncp = %d samples\n',Ncp);
fprintf('CP duration = %.6f ns\n',Tcp*1e9);
fprintf('CP-supported range = %.6f m\n',cpSupportedRange);

for q = 1:Q
    fprintf('Target %d: nominal range = %.6f m, propagation delay = %.6f ns, delay index = %d, grid range = %.6f m, error = %.3e m\n',q,targetRange(q),targetPropagationDelay(q)*1e9,targetDelaySample(q),targetRangeGrid(q),targetRangeError(q));
end

if Ncp<=maxTargetDelaySample
    error('Ncp must be larger than the maximum target delay index.');
end

%% OFDM modulation basis U = F_N^H
FFTmatrix = exp(-1j*2*pi*(0:N-1).'*(0:N-1)/N)/sqrt(N);
U = FFTmatrix';

%% Range axis of the length-N periodic matched-filter output
delaySample = (0:lengthBlock-1).';
rangeAxis = delaySample*c*Ts/2;
plotIndex = rangeAxis>=0 & rangeAxis<=35;

%% Preserve the complete Monte Carlo arrays
SimRangeProfile = zeros(Iter,lengthBlock);
SimTargetProfile = zeros(Iter,lengthBlock,Q);

for ii = 1:Iter
    %% Random 16-PSK symbols and OFDM modulation
    symbolIndex = randi([0,Order-1],N,1);
    s = pskmod(symbolIndex,Order,pi/Order);
    x = U*s;

    %% For L = 1, the sampled Nyquist waveform is xTilde = x
    xTilde = x;

    %% Add CP directly to the time-domain sequence x
    xCP = [x(end-Ncp+1:end);x];

    %% Physical sensing channel: each target produces a linear delay
    lengthYSensing = length(xCP)+maxTargetDelaySample;
    ysTargetFull = complex(zeros(lengthYSensing,Q));

    for q = 1:Q
        startIndex = targetDelaySample(q)+1;
        endIndex = targetDelaySample(q)+length(xCP);
        ysTargetFull(startIndex:endIndex,q) = gamma(q)*xCP;
    end

    ysNoiselessFull = sum(ysTargetFull,2);

    %% Remove CP and select the length-N useful sensing block
    usefulIndex = Ncp+1:Ncp+N;
    ysNoiseless = ysNoiselessFull(usefulIndex);
    ysTarget = ysTargetFull(usefulIndex,:);

    %% Verify that CP converts every physical linear delay to a circular shift
    if ii==1
        for q = 1:Q
            ysTargetEquivalent = gamma(q)*circshift(xTilde,targetDelaySample(q));
            circularizationError = norm(ysTarget(:,q)-ysTargetEquivalent)/max(norm(ysTargetEquivalent),eps);
            fprintf('Target %d circularization error = %.3e\n',q,circularizationError);
        end
    end

    %% Add complex Gaussian noise to the useful sensing block
    signalPower = mean(abs(ysNoiseless).^2);
    noisePower = signalPower/10^(SNRdB/10);
    noise = sqrt(noisePower/2)*(randn(lengthBlock,1)+1j*randn(lengthBlock,1));
    ys = ysNoiseless+noise;

    %% Periodic sensing matched filtering using xTilde = x
    for q = 1:Q
        yMatchedTarget = ifft(fft(ysTarget(:,q)).*conj(fft(xTilde)));
        SimTargetProfile(ii,:,q) = abs(yMatchedTarget.').^2;
    end

    yMatched = ifft(fft(ys).*conj(fft(xTilde)));
    SimRangeProfile(ii,:) = abs(yMatched.').^2;
end

%% Ensemble average of the squared matched-filter output
AveRangeProfile = mean(SimRangeProfile,1);
AveTargetProfile = squeeze(mean(SimTargetProfile,1));
normalization = max(AveRangeProfile);
rangeProfile_dB = 10*log10(AveRangeProfile/normalization+eps);
targetProfile_dB = 10*log10(AveTargetProfile/normalization+eps);

%% Plot Fig. 7
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

plot(rangeAxis(plotIndex),targetProfile_dB(plotIndex,1),':','Color','#F65314','LineWidth',linewidth);
hold on;
plot(rangeAxis(plotIndex),targetProfile_dB(plotIndex,2),':','Color','#00A1F1','LineWidth',linewidth);
plot(rangeAxis(plotIndex),targetProfile_dB(plotIndex,3),':','Color','#8A2BE2','LineWidth',linewidth);
plot(rangeAxis(plotIndex),rangeProfile_dB(plotIndex),'-','Color','#A9A9A9','LineWidth',1.5);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend('Target 1','Target 2','Target 3','Range Profile','Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('Range [m]','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Amplitude [dB]','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([0,35]);
ylim([-55,0]);
xticks(0:5:35);
yticks(-80:10:0);
grid on;
set(gca,'GridLineStyle','--','Gridalpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);

% print(gcf,'Fig7_JSAC_CP_L1.pdf','-dpdf','-vector');