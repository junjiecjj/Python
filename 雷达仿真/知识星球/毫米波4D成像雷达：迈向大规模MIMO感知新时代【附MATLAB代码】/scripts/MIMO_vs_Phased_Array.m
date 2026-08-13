clear
close all
clc

%% FMCW - TDM MIMO
fc              = 77e9;
c               = 3e8;
lambda          = c/fc;
bw              = 150e6;
Nr              = 4;
Nt              = 4;
dr              = lambda/2;
dt              = Nr * lambda/2;
theta           = linspace(-180,180,361);

%% Beampattern Analysis

desiredAngle    = 25;

kTx             = 2 * dt/(lambda);
angleSTVTx      = exp(-1i * pi * kTx *(0:Nt-1) * sind(desiredAngle));
angleSTVTx      = angleSTVTx(:);

kRx             = 2 * dr/(lambda);
angleSTVRx      = exp(-1i * pi * kRx *(0:Nr-1) * sind(desiredAngle));
angleSTVRx      = angleSTVRx(:);

TxRxSRVVA       = kron(angleSTVTx,angleSTVRx);
kVA             = 1;

%% Beampattern 

% - MIMO
AFTxMIMO        = zeros(length(theta),1);
AFRxMIMO        = zeros(length(theta),1);
AFVA            = zeros(length(theta),1);

for i = 1 : length(theta)
    wRxMIMO = exp(-1i * pi * kRx *(0:Nr-1) * sind(theta(i)));
    wRxMIMO = wRxMIMO(:);
    AFRxMIMO(i) = abs(wRxMIMO' * angleSTVRx);
    
    wTxMIMO = exp(-1i * pi * kTx *(1-1) * sind(theta(i)));
    AFTxMIMO(i) = abs(wTxMIMO' * angleSTVTx(1));
    
    wVA = exp(-1i * pi * kVA *(0:Nr*Nt-1) * sind(theta(i)));
    wVA = wVA(:);
    AFVA(i) = 1/Nt * abs(wVA' * TxRxSRVVA); 
end

% - Standard Phased Array
AFRxPA          = zeros(length(theta),1);
AFTxPA          = zeros(length(theta),1);

for i = 1 : length(theta)
    wRxPA = exp(-1i * pi * kRx *(0:Nr-1) * sind(theta(i)));
    wRxPA = wRxPA(:);
    AFRxPA(i) = abs(wRxPA' * angleSTVRx);
    
    wTxPA = exp(-1i * pi * kTx *(0:Nt-1) * sind(theta(i)));
    wTxPA = wTxPA(:);
    AFTxPA(i) = abs(wTxPA' * angleSTVTx); 
end

%%

txPos = (0:Nr-1) * dt;
rxPos = (0:Nt-1) * dr;

%%  Plot Results

% - MIMO
ScreenSize      = get(0,'ScreenSize');
fhdl(1)         = figure('Name','MIMO Beampattern','NumberTitle','off', ...
    'Position', [0 0 floor(ScreenSize(3)/4) floor(ScreenSize(4)/3)]);
movegui('west')
plot(theta,20*log10(AFTxMIMO+0.0001),'-d','MarkerIndices',1:10:length(theta),'markersize',8,'LineWidth',2), hold on
plot(theta,20*log10(AFRxMIMO+0.0001),'-o','MarkerIndices',1:10:length(theta),'markersize',8,'LineWidth',2)
plot(theta,20*log10(AFVA+0.0001),'-s','MarkerIndices',1:10:length(theta),'markersize',8,'LineWidth',2)

xlabel('Angle (degree $^\circ$)','FontSize',11,'Interpreter','Latex')
ylabel('Amplitude (dB)','FontSize',11,'Interpreter','Latex')
legend('Tx Beampattern', 'Rx Beampattern', 'Virtual Array Beampattern', 'Interpreter','Latex', 'FontSize', 11, 'location', 'sw')
grid minor
box on
xlim([-90, 90])
xticks(-90:30:90)

% - Standard Phased Array
fhdl(2)         = figure('Name','Standard Phased Array Beampattern','NumberTitle','off', ...
    'Position', [0 0 floor(ScreenSize(3)/4) floor(ScreenSize(4)/3)]);
movegui('east')
plot(theta,20*log10(AFTxPA+0.0001),'-d','MarkerIndices',1:10:length(theta),'markersize',8,'LineWidth',2), hold on
plot(theta,20*log10(AFRxPA+0.0001),'-o','MarkerIndices',1:10:length(theta),'markersize',8,'LineWidth',2)
plot(theta,20*log10(AFRxPA .* AFTxPA +0.0001),'-s','MarkerIndices',1:10:length(theta),'markersize',8,'LineWidth',2)

xlabel('Angle (degree $^\circ$)','FontSize',11,'Interpreter','Latex')
ylabel('Amplitude (dB)','FontSize',11,'Interpreter','Latex')
legend('Tx Beampattern', 'Rx Beampattern', 'Two-Way Beampattern', 'Interpreter','Latex', 'FontSize', 11, 'location', 'sw')
grid minor
box on
xlim([-90, 90])
xticks(-90:30:90)


fhdl(3)         = figure('Name','Tx-Rx Pos','NumberTitle','off', ...
    'Position', [0 0 floor(ScreenSize(3)/2) floor(ScreenSize(4)/6)]);
movegui('north')
plot(txPos/lambda*2,ones(size(txPos)), 'd','markersize',6,'LineWidth',2), hold on
plot(rxPos/lambda*2,2*ones(size(rxPos)), 'o','markersize',6,'LineWidth',2)

xlabel('Horizontal (Half Wavelength)','Interpreter','latex','FontSize',12')
legend('Tx Chain', 'Rx Chain', 'Interpreter','latex','Location','best','FontSize',11)
grid on
% xlim([0 max(max(xTx), max(xRx))+1])
ylim([0.8 2.2])
yticks([1 2])
yticklabels({'Tx', 'Rx'})
xticks(0:12)