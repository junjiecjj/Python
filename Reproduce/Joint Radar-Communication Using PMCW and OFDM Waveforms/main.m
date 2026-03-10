
% https://ww2.mathworks.cn/help/phased/ug/joint-radar-communication-using-pmcw-and-ofdm-waveforms.html

clc;
clear all;
close all;
% Set the random number generator for reproducibility
rng('default');

fc = 24e9;                              % Carrier frequency (Hz)
B = 100e6;                              % Bandwidth (Hz)

Pt = 0.01;                              % Peak power (W)
Gtx = 20;                               % Tx antenna gain (dB)

jrcpos = [0; 0; 0];                     % JRC position
jrcvel = [0; 0; 0];                     % JRC velocity

jrcmotion = phased.Platform('InitialPosition', jrcpos, 'Velocity', jrcvel);

Grx = 20;                               % Radar Rx antenna gain (dB)
NF = 2.9;                               % Noise figure (dB)
Tref = 290;                             % Reference temperature (K)

Rmax = 200;                             % Maximum range of interest
vrelmax = 60;                           % Maximum relative velocity

tgtpos = [110 45 80; -10 5 -4; 0 0 0];  % Target positions
tgtvel = [-15 40 -32; 0 0 0; 0 0 0];    % Target velocities
tgtmotion = phased.Platform('InitialPosition', tgtpos, 'Velocity', tgtvel);

tgtrcs = [4.7 3.1 2.3];                 % Target RCS
target = phased.RadarTarget('Model', 'Swerling1', 'MeanRCS', tgtrcs, 'OperatingFrequency', fc);

userpos = [100; 20; 0];                  % Downlink user position                   

helperPlotJRCScenario(jrcpos, tgtpos, userpos, tgtvel);

%% JRC System Using a PMCW Waveform
% Generate an m-sequence. The sequence lengths has to be 2^p-1, where p is
% an integer.
p = 8;
prbs = helperMLS(p);
Nprbs = numel(prbs);                    % Number of chips in PRBS

Tchip = 1/B;                             % Chip duration
Tprbs = Nprbs * Tchip;                   % Modulation period

Mpmcw = 256;                             % Number of transmitted data bits 
dataTx = randi([0, 1], [Mpmcw 1]);       % Binary data
bpskTx = pskmod(dataTx, 2);              % BPSK symbols

% Transmit waveform
xpmcw = [prbs * ones(1, Mpmcw); prbs * bpskTx.'];

Tpmcw = 2*Tprbs; 

Tpmcw*Mpmcw;

%% Radar Signal Simulation and Processing

% Set the sample rate equal to the bandwidth
fs = B;

% Setup the transmitter and the radiator
transmitter = phased.Transmitter('Gain', Gtx, 'PeakPower', Pt);

% Assume the JRC is using an isotropic antenna
ant = phased.IsotropicAntennaElement;
radiator = phased.Radiator('Sensor', ant, 'OperatingFrequency', fc);

% Setup the collector and the receiver
collector = phased.Collector('Sensor', ant, 'OperatingFrequency', fc);
receiver = phased.ReceiverPreamp('SampleRate', fs, 'Gain', Grx, 'NoiseFigure', NF, 'ReferenceTemperature', Tref);

% Setup a free space channel to model the JRC signal propagation from the
% transmitter to the targets and back to the radar receiver
radarChannel = phased.FreeSpace('SampleRate', fs, 'TwoWayPropagation', true, 'OperatingFrequency', fc);

% Preallocate space for the signal received by the radar receiver
ypmcwr = zeros(size(xpmcw));

% Transmit one PMCW block at a time assuming that all Mpmcw blocks are
% transmitted within a single CPI
for m = 1:Mpmcw
    % Update sensor and target positions
    [jrcpos, jrcvel] = jrcmotion(Tpmcw);
    [tgtpos, tgtvel] = tgtmotion(Tpmcw);
    
    % Calculate the target angles as seen from the transmit array
    [tgtrng, tgtang] = rangeangle(tgtpos, jrcpos);

    % Transmit signal
    txsig = transmitter(xpmcw(:, m));

    % Radiate signal towards the targets
    radtxsig = radiator(txsig, tgtang);

    % Apply free space channel propagation effects
    chansig = radarChannel(radtxsig, jrcpos, tgtpos, jrcvel, tgtvel); 

    % Reflect signal off the targets
    tgtsig = target(chansig, false);
    
    % Receive target returns at the receive array
    rxsig = collector(tgtsig, tgtang);
 
    % Add thermal noise at the receiver
    ypmcwr(:, m) = receiver(rxsig);
end

ypmcwr1 = ypmcwr(Nprbs+1:end, :) .* (bpskTx.');
ypmcwr1 = ypmcwr1 + ypmcwr(1:Nprbs, :);

% Frequency-domain matched filtering
Ypmcwr = fft(ypmcwr1);
Sprbs = fft(prbs);
Zpmcwr = Ypmcwr .* conj(Sprbs);

% Range-Doppler response object computes DFT in the slow-time domain and
% then IDFT in the fast-time domain to obtain the range-Doppler map
rdr = phased.RangeDopplerResponse('RangeMethod', 'FFT', 'SampleRate', fs, 'SweepSlope', -B/Tprbs,...
    'DopplerOutput', 'Speed', 'OperatingFrequency', fc, 'PRFSource', 'Property', 'PRF', 1/Tpmcw,...
    'ReferenceRangeCentered', false);

figure;
plotResponse(rdr, Zpmcwr, 'Unit', 'db');
xlim([-vrelmax vrelmax]);
ylim([0 Rmax]);

%% Communication Signal Simulation and Processing
% Line-of-sight between the JRC transmitter and the downlink user
dlos = vecnorm(jrcpos - userpos);
taulos = dlos/physconst('LightSpeed');

% Delays and gains due to different path in the communication channel
pathDelays = taulos * [1 1.6 1.1 1.45] - taulos;
averagePathGains = [0 -1 2 -1.5];

% Maximum two-way Doppler shift
fdmax = 2 * speed2dop(vrelmax, freq2wavelen(fc));
commChannel = comm.RicianChannel('PathGainsOutputPort', true, 'DirectPathDopplerShift', 0,... 
    'MaximumDopplerShift', fdmax/2, 'PathDelays', pathDelays, 'AveragePathGains', averagePathGains, ...
    'SampleRate', fs);

% Pass the transmitted signal through the channel
ypmcwc = commChannel(xpmcw(:));

SNRu = 40;                               % SNR at the downlink user's receiver 
ypmcwc = awgn(ypmcwc, SNRu, "measured");
ypmcwc = reshape(ypmcwc, 2*Nprbs, []);

ysound = ypmcwc(1:Nprbs, :); 
Hpmcw = fft(ysound)./Sprbs;

% Compute the frequency domain representation of the received signal
Ypmcwc = fft(ypmcwc(Nprbs + 1:end, :));

% Multiply by the complex-conjugate version of the DFT of the used PRBS
Zpmcwc = Ypmcwc .* conj(Sprbs);

Zpmcwc = Zpmcwc./Hpmcw;

zpmcwc = ifft(Zpmcwc);

[~, idx] = max(abs(zpmcwc), [], 'linear');
bpskRx = zpmcwc(idx).';
bpskRx = bpskRx./Nprbs;
dataRx = pskdemod(bpskRx, 2);

refconst = pskmod([0 1], 2);
constellationDiagram = comm.ConstellationDiagram('NumInputPorts', 1, ...
    'ReferenceConstellation', refconst, 'ChannelNames', {'Received PSK Symbols'});
constellationDiagram(bpskRx(:));

[numErr, ratio] = biterr(dataTx, dataRx)

Nsc = 1024;                              % Number of subcarriers
df = B/Nsc;                              % Separation between OFDM subcarriers
Tsym = 1/df;                             % OFDM symbol duration

df > 10*fdmax

Tcp = range2time(Rmax);                  % Duration of the cyclic prefix (CP)
Ncp = ceil(fs*Tcp);                      % Length of the CP in samples
Tcp = Ncp/fs;                            % Adjust duration of the CP to have an integer number of samples

Tofdm = Tsym + Tcp;                      % OFDM symbol duration with CP
Nofdm = Nsc + Ncp;                       % Number of samples in one OFDM symbol

nullIdx = [1:9 (Nsc/2+1) (Nsc-8:Nsc)]';  % Guard bands and DC subcarrier
Nscd = Nsc-length(nullIdx);              % Number of data subcarriers

bps = 6;                                 % Bits per QAM symbol (and OFDM data subcarrier)
K = 2^bps;                               % Modulation order
Mofdm = 128;                             % Number of transmitted OFDM symbols
dataTx = randi([0,1], [Nscd*bps Mofdm]);
qamTx = qammod(dataTx, K, 'InputType', 'bit', 'UnitAveragePower', true);
ofdmTx = ofdmmod(qamTx, Nsc, Ncp, nullIdx);

%% Radar Signal Simulation and Processing
xofdm = reshape(ofdmTx, Nofdm, Mofdm);

% OFDM waveform is not a constant modulus waveform. The generated OFDM
% samples have power much less than one. To fully utilize the available
% transmit power, normalize the waveform such that the sample with the
% largest power had power of one.
xofdm = xofdm/max(sqrt(abs(xofdm).^2), [], 'all');

% Preallocate space for the signal received by the radar receiver
yofdmr = zeros(size(xofdm));

% Reset the platform objects representing the JRC and the targets before
% running the simulation
reset(jrcmotion);
reset(tgtmotion);

% Transmit one OFDM symbol at a time assuming that all Mofdm symbols are
% transmitted within a single CPI
for m = 1:Mofdm
    % Update sensor and target positions
    [jrcpos, jrcvel] = jrcmotion(Tofdm);
    [tgtpos, tgtvel] = tgtmotion(Tofdm);
    
    % Calculate the target angles as seen from the transmit array
    [tgtrng, tgtang] = rangeangle(tgtpos, jrcpos);

    % Transmit signal
    txsig = transmitter(xofdm(:, m));

    % Radiate signal towards the targets
    radtxsig = radiator(txsig, tgtang);

    % Apply free space channel propagation effects
    chansig = radarChannel(radtxsig, jrcpos, tgtpos, jrcvel, tgtvel); 

    % Reflect signal off the targets
    tgtsig = target(chansig, false);
    
    % Receive target returns at the receive array
    rxsig = collector(tgtsig, tgtang);
 
    % Add thermal noise at the receiver
    yofdmr(:, m) = receiver(rxsig);
end


yofdmr1 = reshape(yofdmr, Nofdm*Mofdm, 1);

% Demodulate the received OFDM signal (compute DFT)
Yofdmr = ofdmdemod(yofdmr1, Nsc, Ncp, Ncp, nullIdx);

Zofdmr = Yofdmr./qamTx;

% Range-Doppler response object computes DFT in the slow-time domain and
% then IDFT in the fast-time domain to obtain the range-Doppler map
rdr = phased.RangeDopplerResponse('RangeMethod', 'FFT', 'SampleRate', fs, 'SweepSlope', -B/Tofdm,...
    'DopplerOutput', 'Speed', 'OperatingFrequency', fc, 'PRFSource', 'Property', 'PRF', 1/Tofdm, ...
    'ReferenceRangeCentered', false);

figure;
plotResponse(rdr, Zofdmr, 'Unit', 'db');
xlim([-vrelmax vrelmax]);
ylim([0 Rmax]);

[yofdmc, pathGains] = commChannel(ofdmTx);
yofdmc = awgn(yofdmc, SNRu, "measured");
channelInfo = info(commChannel);
pathFilters = channelInfo.ChannelFilterCoefficients;
toffset = channelInfo.ChannelFilterDelay;
Hofdm = ofdmChannelResponse(pathGains, pathFilters, Nsc, Ncp, ...
    setdiff(1:Nsc, nullIdx), toffset);

zeropadding = zeros(toffset, 1);
ofdmRx = [yofdmc(toffset+1:end,:); zeropadding];

qamRx = ofdmdemod(ofdmRx, Nsc, Ncp, Ncp/2, nullIdx);
qamEq = ofdmEqualize(qamRx, Hofdm(:), 'Algorithm', 'zf');

dataRx = qamdemod(qamEq, K, 'OutputType', 'bit', 'UnitAveragePower', true);

refconst = qammod(0:K-1, K, 'UnitAveragePower', true);
constellationDiagram = comm.ConstellationDiagram('NumInputPorts', 1, ...
    'ReferenceConstellation', refconst, 'ChannelNames', {'Received QAM Symbols'});

% Show QAM data symbols comprising the first 10 received OFDM symbols
constellationDiagram(qamEq(1:Nscd*10).');

[numErr,ratio] = biterr(dataTx, dataRx);

