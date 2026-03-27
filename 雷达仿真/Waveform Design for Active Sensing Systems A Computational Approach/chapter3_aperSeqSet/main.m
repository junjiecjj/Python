%% Part A: CAN & Khan
N = 40; M = 3;

Xkhan = khan40by3;
load canX_40by3.mat;
Xcan = canX_40by3; %Xcan = canmimo(N,M);

[r_khan ISL_khan PSL_khan] = aperplotmimo(Xkhan);
[r_can ISL_can PSL_can] = aperplotmimo(Xcan);
disp('************ fit error *************');
disp(['Khan: ' num2str(ISL_khan/(M*N^2))]); 
disp(['CAN:  ' num2str(ISL_can/(M*N^2))]); 

for m1 = 1:M
    for m2 = 1:M
        r_khan = aperplotcross(Xkhan(:,m1), Xkhan(:,m2));
        r_can = aperplotcross(Xcan(:,m1), Xcan(:,m2));
        figure;
        plot(-N+1:N-1, abs(r_khan)/N, 'r--', -N+1:N-1, abs(r_can)/N, 'b-');
        xlabel('k'); ylabel(sprintf('         |r_{%d%d}(k)|', m1, m2));
        legend('CE', 'Multi-CAN');
        axis([-N+1 N-1 0 1]);
        myboldify;
        myresize(sprintf('MIMOCAN_r%d%d', m1, m2));        
    end
end

%% Part A: CAN vs. Hadamard+PN
clear; clc;
NAll = 2.^((7:13)');
M = 3;
lengthNum = length(NAll);

peakAuto = zeros(lengthNum, 3);
peakCross = zeros(lengthNum, 3);
fitError = zeros(lengthNum, 3);

Xhadamard = zeros(NAll(end), M, lengthNum);
Xcan = zeros(NAll(end), M, lengthNum);
Xrand = zeros(NAll(end), M, lengthNum);

for k = 1:lengthNum
    disp(['sequence length: ' num2str(NAll(k))]);
    N = NAll(k);
    Xhadamard(1:N,:,k) = orthoghadamard(N,M);
    [peakAuto(k,1) peakCross(k,1) fitError(k,1)] = mimocriterion(Xhadamard(1:N,:,k));
    [Xcan(1:N,:,k) Xrand(1:N,:,k)] = canmimoMC(N,M);
    [peakAuto(k,2) peakCross(k,2) fitError(k,2)] = mimocriterion(Xcan(1:N,:,k));
    [peakAuto(k,3) peakCross(k,3) fitError(k,3)] = mimocriterion(Xrand(1:N,:,k));
end

figure;
semilogx(NAll, 20*log10(peakAuto(:,3)), 'm*--', ...
    NAll, 20*log10(peakAuto(:,1)), 'ro--', ...
    NAll, 20*log10(peakAuto(:,2)), 'bd-');
xlabel('N');
ylabel('Auto-Corr Peak Sidelobe (dB)');
legend('random-phase', 'Hadamard+PN', 'Multi-CAN');
axis([NAll(1) NAll(end) -35 -10]);
set(gca,'XTick',NAll)
set(gca,'XTickLabel',{'128','256','512','1024','2048','4096','8192'})
myboldify;
myresize('MIMOCAN_autoPeak');
    
figure;
semilogx(NAll, 20*log10(peakCross(:,3)), 'm*--', ...
    NAll, 20*log10(peakCross(:,1)), 'ro--', ...
    NAll, 20*log10(peakCross(:,2)), 'bd-');
xlabel('N');
ylabel('Cross-Corr Peak (dB)');
legend('random-phase', 'Hadamard+PN', 'Multi-CAN');
axis([NAll(1) NAll(end) -35 -10]);
set(gca,'XTick',NAll)
set(gca,'XTickLabel',{'128','256','512','1024','2048','4096','8192'})
myboldify;
myresize('MIMOCAN_crossPeak');

figure;
semilogx(NAll, fitError(:,3), 'm*--', NAll, fitError(:,1), 'ro--', ...
    NAll, fitError(:,2), 'bd-');
xlabel('N');
ylabel('Normalized Fitting Error');
legend('random-phase', 'Hadamard+PN', 'Multi-CAN');
axis([NAll(1) NAll(end) 1.5 3.5]);
set(gca,'XTick',NAll)
set(gca,'XTickLabel',{'128','256','512','1024','2048','4096','8192'})
myboldify;
myresize('MIMOCAN_fittingError');

% save partA.mat

%% part B: WeCAN and CA, example 1
N = 256; M = 4; P = 50;
Rd = N * eye(M);
X0 = exp(1i * 2*pi * rand(N,M));

tic;
Xcan = canmimo(N, M, X0);
timeCAN = toc;
[peakAutoCAN peakCrossCAN fitErrorCAN] = mimocriterion(Xcan, Rd, P);

tic;
Xca = camimo(N,M,P,X0);
timeCA = toc;
[peakAutoCA peakCrossCA fitErrorCA] = mimocriterion(Xca, Rd, P);
aperplotmimo(Xca, '', [-80 0]);
plot([-(P-1) -(P-1)], [-80 0], 'g--', [P-1 P-1], [-80 0], 'g--'); 
myboldify;
myresize('MIMOWeCAN_ca');

gamma = zeros(N,1); gamma(1:P) = 1;
tic;
Xwecan = wecanmimo(N, M, gamma, X0);
timeWeCAN = toc;
[peakAutoWeCAN peakCrossWeCAN fitErrorWeCAN] = mimocriterion(Xwecan, Rd, P);
aperplotmimo(Xwecan, '', [-80 0]);
plot([-(P-1) -(P-1)], [-80 0], 'g--', [P-1 P-1], [-80 0], 'g--'); 
myboldify;
myresize('MIMOWeCAN_wecan');

disp(['MIMO-CAN (' num2str(timeCAN) ' sec) :' num2str(20*log10(peakAutoCAN))...
    ' ' num2str(20*log10(peakCrossCAN)) ' ' num2str(fitErrorCAN)]);
disp(['CA (' num2str(timeCA) ' sec) :' num2str(20*log10(peakAutoCA))...
    ' ' num2str(20*log10(peakCrossCA)) ' ' num2str(fitErrorCA)]);
disp(['MIMO-WeCAN (' num2str(timeWeCAN) ' sec) :' num2str(20*log10(peakAutoWeCAN))...
    ' ' num2str(20*log10(peakCrossWeCAN)) ' ' num2str(fitErrorWeCAN)]);
% save partB.mat;

%% part C: WeCAN
N = 256; M = 4; P1 = 20; P2 = 237;
Rd = N * eye(M);
X0 = exp(1i * 2*pi * rand(N,M));
gamma = zeros(N,1); 
gamma(1:20) = 1; gamma(237:256) = 1;

tic;
Xcan = canmimo(N, M, X0);
timeCAN = toc;
[peakAutoCAN peakCrossCAN fitErrorCAN] = mimocriterion(Xcan, Rd, gamma);
aperplotmimo(Xcan, '', [-80 0]);
plot([-(P1-1) -(P1-1)], [-80 0], 'g--', [P1-1 P1-1], [-80 0], 'g--');
plot([-(P2-1) -(P2-1)], [-80 0], 'g--', [P2-1 P2-1], [-80 0], 'g--'); 
hold off;
myboldify;
myresize('MIMOWeCAN_can');

tic;
Xwecan = wecanmimo(N, M, gamma, X0);
timeWeCAN = toc;
[peakAutoWeCAN peakCrossWeCAN fitErrorWeCAN] = mimocriterion(Xwecan, Rd, gamma);
aperplotmimo(Xwecan, '', [-80 0]);
plot([-(P1-1) -(P1-1)], [-80 0], 'g--', [P1-1 P1-1], [-80 0], 'g--');
plot([-(P2-1) -(P2-1)], [-80 0], 'g--', [P2-1 P2-1], [-80 0], 'g--'); 
hold off;
myboldify;
myresize('MIMOWeCAN_wecan2');

disp(['CAN (' num2str(timeCAN) ' sec) :' num2str(20*log10(peakAutoCAN))...
    ' ' num2str(20*log10(peakCrossCAN)) ' ' num2str(fitErrorCAN)]);
disp(['WeCAN (' num2str(timeWeCAN) ' sec) :' num2str(20*log10(peakAutoWeCAN))...
    ' ' num2str(20*log10(peakCrossWeCAN)) ' ' num2str(fitErrorWeCAN)]);
%save partC.mat;

%% part E: SAR imaging
NTilde = 20; % number of positions where data are collected

Mt = 4; % number of transmit antennas
Mr = 4; % number of receive antennas
N = 256; % number of subpulses transmitted by each antenna
P = 30; % [0 P-1] is the range of SAR imaging
scanSet = ((-30):30)';  % angle of imaging, K-by-1, in the unit of degree
K = length(scanSet);

rcs = zeros(P, K); % the matrix of radar cross section, P-by-K
rcsTmp = (randn(P,K) + 1i * randn(P,K))/sqrt(2);
rcsTmp = rcsTmp / max(abs(rcsTmp(:)));
index = false(P, K);
index(floor(P/8):floor(7*P/8)-5, floor(K/8)+3) = 1;
index(floor(7*P/8)-5, floor(K/8)+3:floor(3*K/8)+3) = 1;
index(floor(P/8):floor(7*P/8)-5, floor(3*K/8)+3) = 1;
index(floor(P/8)+5:floor(7*P/8), floor(5*K/8)+3) = 1;
index(floor(P/8)+5, floor(5*K/8)+3:floor(7*K/8)+3) = 1;
index(floor(P/2), floor(5*K/8)+3:floor(6*K/8)+3) = 1;
rcs(index) = rcsTmp(index); 

rcsplot(P, scanSet, rcs, 'Ground Truth');
myresize('MIMOWeCAN_sarTruth');

% Hadamard+PN
waveform = orthoghadamard(N, Mt);
rcsEst = rcsestimation(Mt, Mr, N, P, scanSet, rcs, waveform, 'ls', NTilde);
rcsplot(P, scanSet, rcsEst, 'Hadamard+PN (Least Squares)');
myresize('MIMOWeCAN_sarHadamardLS');
rcsEst = rcsestimation(Mt, Mr, N, P, scanSet, rcs, waveform, 'capon', NTilde);
rcsplot(P, scanSet, rcsEst, 'Hadamard+PN (Capon)');
myresize('MIMOWeCAN_sarHadamardCapon');

% WeCAN
load X_256by4_30_WeCAN.mat;
waveform = X_256by4_30_WeCAN;
rcsEst = rcsestimation(Mt, Mr, N, P, scanSet, rcs, waveform, 'ls', NTilde);
rcsplot(P, scanSet, rcsEst, 'Multi-WeCAN (Least Squares)');
myresize('MIMOWeCAN_sarCadLS');
rcsEst = rcsestimation(Mt, Mr, N, P, scanSet, rcs, waveform, 'capon', NTilde);
rcsplot(P, scanSet, rcsEst, 'Multi-WeCAN (Capon)');
myresize('MIMOWeCAN_sarCadCapon');

%% Quantization
load partA.mat;

q = 5; % quantization level
for k = 1:lengthNum
    disp(['sequence length: ' num2str(NAll(k))]);
    N = NAll(k);
    [peakAuto(k,1) peakCross(k,1) fitError(k,1)] = mimocriterion(Xhadamard(1:N,:,k));
    x = Xcan(1:N,:,k);
    x_phase = angle(x);
    index = (x_phase < 0);
    x_phase(index) = x_phase(index) + 2 * pi;
    x = exp(1i * floor(x_phase/(2*pi/(2^q))) * 2*pi/(2^q));
    [peakAuto(k,2) peakCross(k,2) fitError(k,2)] = mimocriterion(x);
    [peakAuto(k,3) peakCross(k,3) fitError(k,3)] = mimocriterion(Xrand(1:N,:,k));
end

figure;
semilogx(NAll, 20*log10(peakAuto(:,3)), 'm*--', ...
    NAll, 20*log10(peakAuto(:,1)), 'ro--', ...
    NAll, 20*log10(peakAuto(:,2)), 'bd-');
xlabel('N');
ylabel('Auto-Corr Peak Sidelobe (dB)');
legend('random-phase', 'Hadamard+PN', 'Quantized Multi-CAN');
axis([NAll(1) NAll(end) -35 -10]);
set(gca,'XTick',NAll)
set(gca,'XTickLabel',{'128','256','512','1024','2048','4096','8192'})
myboldify;
myresize('MIMOCAN_autoPeak_quantize');
    
figure;
semilogx(NAll, 20*log10(peakCross(:,3)), 'm*--', ...
    NAll, 20*log10(peakCross(:,1)), 'ro--', ...
    NAll, 20*log10(peakCross(:,2)), 'bd-');
xlabel('N');
ylabel('Cross-Corr Peak (dB)');
legend('random-phase', 'Hadamard+PN', 'Quantized Multi-CAN');
axis([NAll(1) NAll(end) -35 -10]);
set(gca,'XTick',NAll)
set(gca,'XTickLabel',{'128','256','512','1024','2048','4096','8192'})
myboldify;
myresize('MIMOCAN_crossPeak_quantize');

figure;
semilogx(NAll, fitError(:,3), 'm*--', NAll, fitError(:,1), 'ro--', ...
    NAll, fitError(:,2), 'bd-');
xlabel('N');
ylabel('Normalized Fitting Error');
legend('random-phase', 'Hadamard+PN', 'Quantized Multi-CAN');
axis([NAll(1) NAll(end) 1.5 3.5]);
set(gca,'XTick',NAll)
set(gca,'XTickLabel',{'128','256','512','1024','2048','4096','8192'})
myboldify;
myresize('MIMOCAN_fittingError_quantize');
