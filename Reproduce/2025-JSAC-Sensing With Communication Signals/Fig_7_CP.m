

%% Fig. 7 reproduction with CP added before pulse shaping
% Sensing With Communication Signals: From Information Theory to Signal Processing
% Transmitter: s -> x = U*s -> add CP -> upsample -> RRC pulse shaping
% Sensing channel: physical linear delays of the unique transmitted waveform
% Receiver: select the useful block -> periodic waveform matched filtering

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

%% Discrete waveform parameters
N = 4096;
L = 20;
alpha = 0.35;
span = 20;
rangeSampleSpacing = 0.025;
desiredCPRange = 40;
c = 299792458;

%% Target amplitudes
gamma = 10.^(targetAmplitude_dB/20).*exp(1j*targetPhase);

%% Physical range, propagation delay and sample-index conversion
Fs = c/(2*rangeSampleSpacing);
Ts = 1/Fs;
symbolPeriod = L*Ts;
symbolRangeSpacing = L*rangeSampleSpacing;
targetPropagationDelay = 2*targetRange/c;
targetDelaySample = round(targetPropagationDelay/Ts);
targetRangeGrid = targetDelaySample*c*Ts/2;
targetRangeError = targetRangeGrid-targetRange;
maxTargetDelaySample = max(targetDelaySample);

%% RRC pulse: causal FIR representation used after CP insertion
p = rcosdesign(alpha,span,L,'sqrt').';
p = p/norm(p);
pulseMemorySample = length(p)-1;

%% CP is defined before upsampling, hence its high-rate length is L*Ncp
% It must cover both the pulse-shaping memory and the maximum target delay.
NcpFromDesiredRange = ceil(desiredCPRange/symbolRangeSpacing);
NcpFromTotalMemory = ceil((pulseMemorySample+maxTargetDelaySample)/L);
Ncp = max(NcpFromDesiredRange,NcpFromTotalMemory);
NcpSample = L*Ncp;
Tcp = Ncp*symbolPeriod;
cpSupportedRange = NcpSample*c*Ts/2;
lengthBlock = L*N;

fprintf('Sampling rate Fs = %.6f GHz\n',Fs/1e9);
fprintf('High-rate range grid = %.6f m/sample\n',rangeSampleSpacing);
fprintf('Symbol-rate range grid = %.6f m/symbol\n',symbolRangeSpacing);
fprintf('RRC pulse memory = %d high-rate samples\n',pulseMemorySample);
fprintf('Maximum target delay = %d high-rate samples\n',maxTargetDelaySample);
fprintf('Ncp = %d symbol-rate samples = %d high-rate samples\n',Ncp,NcpSample);
fprintf('CP duration = %.6f ns\n',Tcp*1e9);
fprintf('CP-supported range = %.6f m\n',cpSupportedRange);

for q = 1:Q
    fprintf('Target %d: nominal range = %.6f m, propagation delay = %.6f ns, delay index = %d, grid range = %.6f m, error = %.3e m\n',q,targetRange(q),targetPropagationDelay(q)*1e9,targetDelaySample(q),targetRangeGrid(q),targetRangeError(q));
end

if NcpSample<pulseMemorySample+maxTargetDelaySample
    error('The CP does not cover the pulse-shaping memory plus maximum target delay.');
end

%% Eq. (35): OFDM modulation basis U = F_N^H
FFTmatrix = exp(-1j*2*pi*(0:N-1).'*(0:N-1)/N)/sqrt(N);
U = FFTmatrix';

%% Range axis of the length-LN periodic matched-filter output
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

    %% Add the CP directly to x before upsampling and pulse shaping
    xCP = [x(end-Ncp+1:end);x];

    %% Upsample the CP-extended sequence
    xCPUp = complex(zeros(L*length(xCP),1));
    xCPUp(1:L:end) = xCP;

    %% Physical transmit pulse shaping is a linear convolution
    xTransmit = conv(xCPUp,p,'full');

    %% Select the pulse-shaped useful block after the high-rate CP
    usefulIndex = NcpSample+1:NcpSample+lengthBlock;
    xTilde = xTransmit(usefulIndex);

    %% Verify Eq. (44)-(47): CP makes useful-block pulse shaping periodic
    if ii==1
        xUp = complex(zeros(lengthBlock,1));
        xUp(1:L:end) = x;
        pPeriodic = complex(zeros(lengthBlock,1));
        pPeriodic(1:length(p)) = p;
        xTildeEquivalent = ifft(fft(xUp).*fft(pPeriodic));
        pulseCircularizationError = norm(xTilde-xTildeEquivalent)/max(norm(xTildeEquivalent),eps);
        fprintf('Pulse-shaping circularization error = %.3e\n',pulseCircularizationError);
    end

    %% Physical sensing channel: each target produces a linear delay
    lengthYSensing = length(xTransmit)+maxTargetDelaySample;
    ysTargetFull = complex(zeros(lengthYSensing,Q));

    for q = 1:Q
        startIndex = targetDelaySample(q)+1;
        endIndex = targetDelaySample(q)+length(xTransmit);
        ysTargetFull(startIndex:endIndex,q) = gamma(q)*xTransmit;
    end

    ysNoiselessFull = sum(ysTargetFull,2);

    %% Select the same useful observation interval at the sensing receiver
    ysNoiseless = ysNoiselessFull(usefulIndex);
    ysTarget = ysTargetFull(usefulIndex,:);

    %% Verify that CP converts every physical linear delay to a circular shift
    if ii==1
        for q = 1:Q
            ysTargetEquivalent = gamma(q)*circshift(xTilde,targetDelaySample(q));
            targetCircularizationError = norm(ysTarget(:,q)-ysTargetEquivalent)/max(norm(ysTargetEquivalent),eps);
            fprintf('Target %d circularization error = %.3e\n',q,targetCircularizationError);
        end
    end

    %% Add complex Gaussian noise to the selected useful sensing block
    signalPower = mean(abs(ysNoiseless).^2);
    noisePower = signalPower/10^(SNRdB/10);
    noise = sqrt(noisePower/2)*(randn(lengthBlock,1)+1j*randn(lengthBlock,1));
    ys = ysNoiseless+noise;

    %% Periodic sensing matched filtering using the actual transmitted block
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

%% Plot the three-target range profile
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
ylim([-80,0]);
xticks(0:5:35);
yticks(-80:10:0);
grid on;
set(gca,'GridLineStyle','--','Gridalpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);

% print(gcf,'Fig7_JSAC_CP.pdf','-dpdf','-vector');