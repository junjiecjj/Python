
% https://ww2.mathworks.cn/help/phased/examples.html?s_tid=CRUX_topnav&category=radar-and-wireless-coexistence
% https://ww2.mathworks.cn/help/phased/ug/waveform-design-for-a-dual-function-mimo-radcom-system.html



clc;
clear all;
close all;
addpath('./functions');
rng('default');                     % Set random number generator for reproducibility

fc = 6e9;                           % Carrier frequency (Hz)
Pt = 0.5e6;                         % Peak transmit power (W)
lambda = freq2wavelen(fc);          % Wavelength (m)

N = 16;                             % Number of array elements
d = 0.5*lambda;                     % Array element spacing (m)

% Use isotropic antenna elements
element = phased.IsotropicAntennaElement('BackBaffled', true);
array = phased.ULA('Element', element, 'NumElements', N, 'ElementSpacing', d, 'ArrayAxis', 'y');

% Three targets of interest
tgtAz = [-60 10 45];                % Azimuths of the targets of interest
tgtRng = [5.31e3, 6.23e3, 5.7e3];   % Ranges of the targets of interest

ang = linspace(-90, 90, 200);       % Grid of azimuth angles
beamwidth = 10;                     % Desired beamwidth

% Desired beam pattern
idx = false(size(ang));
for i = 1:numel(tgtAz)
    idx = idx | ang >= tgtAz(i)-beamwidth/2 & ang <= tgtAz(i)+beamwidth/2;
end

Bdes = zeros(size(ang));
Bdes(idx) = 1;

figure(1);
plot(ang, Bdes, 'LineWidth', 2);
xlabel('Azimuth (deg)');
ylabel('Desired Beam Pattern');
title('Desired Beam Pattern');
grid on;
ylim([0 1.1]);

%% Objective of the Communication Component
K = 4;                              % Number of communication users
M = 30;                             % Number of communication symbols

Q = 4;
data = randi([0 Q-1], K, M);        % Binary data
S = pskmod(data, Q, pi/Q);          % QPSK symbols

% User locations are random
txpos = [rand(1, K)*1.5e3; rand(1, K)*2.4e3 - 1.2e3; zeros(1, K)];

% Create a scattering channel matrix assuming 100 independent scatterers
numscat = 100;
rxpos = array.getElementPosition();    
H = scatteringchanmtx(rxpos, txpos, numscat).';

% Normalize the antenna element positions by the wavelength
normalizedPos = rxpos/lambda;

% Solve the optimization problem to find the covariance matrix
Rmmse = helperMMSECovariance(normalizedPos, Bdes, ang);

% helperMMSECovariance returns a covariance matrix with 1s along the
% diagonal such that the total transmit power is equal to N. Renormalize
% the covariance matrix to make the total transmit power equal to Pt.
Rmmse = Rmmse*Pt/N;

% Matrix of steering vectors corresponding to the angles in the grid ang
A = steervec(normalizedPos, [ang; zeros(size(ang))]);

% Compute the resulting beam pattern given the found covariance matrix
Bmmse = abs(diag(A'*Rmmse*A))/(4*pi);

figure(2);
hold on
plot(ang, pow2db(Bdes + eps), 'LineWidth', 2)
plot(ang, pow2db(Bmmse/max(Bmmse)), 'LineWidth', 2)

grid on
xlabel('Azimuth (deg)')
ylabel('(dB)');
legend('Desired', 'MMSE Covariance', 'Location', 'southoutside', 'Orientation', 'horizontal');
ylim([-30 1]);
title('Transmit Beam Pattern');


%% From Covariance Matrix to Waveform
eta = 1.1;                          % Parameter that controls low PAR constraint

% Find a set of waveform with the covariance equal to Rmmse using the
% cyclic algorithm. The length of each waveform is M.
Xca = helperCAWaveformSynthesis(Rmmse, M, eta);

% Covariance matrix of the computed waveforms
Rca = Xca*Xca'/M;

% The resulting beam pattern
Bca = abs(diag(A'*Rca*A))/(4*pi);

figure(3);
hold on;
plot(ang, pow2db(Bdes + eps), 'LineWidth', 2);
plot(ang, pow2db(Bmmse/max(Bmmse)), 'LineWidth', 2);
plot(ang, pow2db(Bca/max(Bca)), 'LineWidth', 2);

grid on;
xlabel('Azimuth (deg)');
ylabel('(dB)');
legend('Desired', 'MMSE Covariance', 'Radar Waveforms', 'Location', 'southoutside', 'Orientation', 'horizontal')
ylim([-30 1]);
title('Transmit Beam Pattern');

% Average power
Pn = diag(Rca);

% Peak-to-average power ratio
PARn = max(abs(Xca).^2, [], 2)./Pn;

array2table([Pn.'; PARn.'], 'VariableNames', compose('%d', 1:N), 'RowNames', {'Average Power', 'Peak-to-Average Power Ratio'})

% Plot the autocorrelation function for the nth waveform in Xca
n = 1;
[acmag_ca, delay] = ambgfun(Xca(n, :), 1, 1/M, "Cut", "Doppler");
acmag_ca_db = mag2db(acmag_ca);
psl_ca = sidelobelevel(acmag_ca_db);

figure(4);
ax = gca;
colors = ax.ColorOrder;
plot(delay, acmag_ca_db, 'Color', colors(3, :), 'Linewidth', 2);
yline(psl_ca,'Label',sprintf('Sidelobe Level (%.2f dB)', psl_ca));
xlabel('Lag (subpulses)');
ylabel('(dB)');
title(sprintf('Autocorrelation Function for Waveform #%d', n));
grid on;
ylim([-45 0]);


%% Radar-Communication Waveform Synthesis
% Radar-communication tradeoff parameter
rho = 0.4;

% Use RCG to embed the communication symbols in S into the waveforms in Xca
% assuming known channel matrix H.
Xradcom = helperRadComWaveform(H, S, Xca, Pt, rho);

% Compute the corresponding waveform covariance matrix and the transmit beam pattern
Rradcom = Xradcom*Xradcom'/M;
Bradcom = abs(diag(A'*Rradcom*A))/(4*pi);

figure(5);
hold on;
plot(ang, pow2db(Bdes + eps), 'LineWidth', 2);
plot(ang, pow2db(Bmmse/max(Bmmse)), 'LineWidth', 2);
plot(ang, pow2db(Bca/max(Bca)), 'LineWidth', 2);
plot(ang, pow2db(Bradcom/max(Bradcom)), 'LineWidth', 2);

grid on;
xlabel('Azimuth (deg)');
ylabel('(dB)');
legend('Desired', 'MMSE Covariance', 'Radar Waveforms', 'RadCom Waveforms',...
    'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2)
ylim([-30 1])
title('Transmit Beam Pattern');


[acmag_rc, delay] = ambgfun(Xradcom(n, :), 1, 1/M, "Cut", "Doppler");
acmag_rc_db = mag2db(acmag_rc);
psl_rc = sidelobelevel(acmag_rc_db);

figure(6);
ax = gca;
colors = ax.ColorOrder;

hold on;
plot(delay, acmag_ca_db, 'Color', colors(3, :), 'LineWidth', 2)
plot(delay, acmag_rc_db, 'Color', colors(4, :), 'LineWidth', 2)
yline(psl_rc,'Label',sprintf('Sidelobe Level (%.2f dB)', psl_rc))
xlabel('Lag (subpulses)')
ylabel('(dB)')
title(sprintf('Autocorrelation Function for Waveform #%d', n))
grid on;
ylim([-30 0]);

legend('Radar Waveform', 'RadCom Waveform');

%% Joint Radar-Communication Simulation
tau = 5e-7;                         % Subpulse duration
B = 1/tau;                          % Bandwidth
prf = 15e3;                         % PRF
fs = B;                             % Sample rate

% The RadCom system is located at the origin and is not moving
radcompos = [0; 0; 0];
radcomvel = [0; 0; 0];

t = 0:1/fs:(1/prf);

waveform = zeros(numel(t), N);
waveform(1:M, :) = Xradcom' / sqrt(Pt/N);

transmitter = phased.Transmitter('Gain', 0, 'PeakPower', Pt/N, 'InUseOutputPort', true);
txsig = zeros(size(waveform));
for n = 1:N
    txsig(:, n) = transmitter(waveform(:, n));
end

% Positions of the targets of interest in Cartesian coordinates
[x, y, z] = sph2cart(deg2rad(tgtAz), zeros(size(tgtAz)), tgtRng);
tgtpos = [x; y; z];
% Assume the targets are static
tgtvel = zeros(3, numel(tgtAz));

% Calculate the target angles as seen from the transmit array
[tgtRng, tgtang] = rangeangle(tgtpos, radcompos);

radiator = phased.Radiator('Sensor', array, 'OperatingFrequency', fc, 'CombineRadiatedSignals', true);
radtxsig = radiator(txsig, tgtang);

channel = phased.FreeSpace('SampleRate', fs, 'TwoWayPropagation', true, 'OperatingFrequency',fc);
radtxsig = channel(radtxsig, radcompos, tgtpos, radcomvel, tgtvel);

% Target radar cross sections
tgtRCS = [2.7 3.1 4.3];

% Reflect pulse off targets
target = phased.RadarTarget('Model', 'Nonfluctuating', 'MeanRCS', tgtRCS, 'OperatingFrequency', fc);
tgtsig = target(radtxsig);

% Receive target returns at the receive array
collector = phased.Collector('Sensor', array, 'OperatingFrequency', fc);
rxsig = collector(tgtsig, tgtang);

receiver = phased.ReceiverPreamp('Gain', 0, 'NoiseFigure', 2.7, 'SampleRate', fs);
rxsig = receiver(rxsig);

% Compute range-angle response
rngangresp = phased.RangeAngleResponse(...
    'SensorArray',array,'OperatingFrequency',fc,...
    'SampleRate',fs,'PropagationSpeed',collector.PropagationSpeed);

resp = zeros(numel(t), rngangresp.NumAngleSamples, N);
% Apply matched filter N times. One time for each waveform. Then integrate the results.
for i = 1:N
    [resp(:, :, i), rng_grid, ang_grid] = rngangresp(rxsig, flipud(Xradcom(i, :).'));
end

% Plot the range angle response
figure(7);
resp_int = sum(abs(resp).^2, 3);
resp_max = max(resp_int, [], 'all');
imagesc(ang_grid, rng_grid, pow2db(resp_int/resp_max))
clim([-20 1])
axis xy
xlabel('Azimuth (deg)')
ylabel('Range (m)')
title('Range-Angle Response')
cbar = colorbar;
cbar.Label.String = '(dB)';

%% transmit the dual-function waveforms through the MIMO scattering channel H and compute the resulting error rate.
rd = pskdemod(H*Xradcom, Q, pi/Q);
[numErr, errRatio] = symerr(data, rd)

% Vary the radar-communication tradeoff parameter from 0 to 1
rho = 0.0 : 0.1 : 1;

er = zeros(size(rho));
psl_ca = zeros(size(rho));
bpse = zeros(size(rho));

for i = 1:numel(rho)
    Xrc = helperRadComWaveform(H, S, Xca, Pt, rho(i));

    % Transmit through the communication channel and compute the error rate
    rd = pskdemod(H*Xrc, Q, pi/Q);
    [~, er(i)] = symerr(data, rd);

    % Compute the peak to sidelobe level
    psl_ca(i) = helperAveragePSL(Xrc);
    
    % Compute the beam pattern
    Rrc = Xrc*Xrc'/M;
    Brc = abs(diag(A'*Rrc*A).')/(4*pi);

    % Squared error between the desired beam pattern and the beam pattern produced by the RadCom waveforms Xrc
    bpse(i) = trapz(deg2rad(ang), (Bdes - Brc/max(Brc)).^2.*cosd(ang));
end

figure(8);
tiledlayout(3, 1);

ax = nexttile;
semilogy(rho, er, 'LineWidth', 2);
xlim([0 1]);
ylim([1e-3 1]);
grid on;
title('Symbol Error Rate');

nexttile;
plot(rho, psl_ca, 'LineWidth', 2);
ylabel('(dB)');
grid on;
title({'Peak to Sidelobe Level', '(averaged over 16 waveforms)'});

nexttile;
plot(rho, pow2db(bpse), 'LineWidth', 2);
ylabel('(dB)');
grid on;
title('Squared Error Between the Desired and the RadCom Beam Patterns');

xlabel('\rho');