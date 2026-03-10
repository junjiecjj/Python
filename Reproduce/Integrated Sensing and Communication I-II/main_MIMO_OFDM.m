% https://ww2.mathworks.cn/help/phased/ug/integrated-sensing-and-communication-2-communication-centric-approach-using-mimo-ofdm.html
clc;
clear all;
close all;
addpath('./functions');

% Set the random number generator for reproducibility
rng('default');

carrierFrequency = 6e9;                                                 % Carrier frequency (Hz)
waveLength = freq2wavelen(carrierFrequency);                            % Wavelength
bandwidth = 100e6;                                                      % Bandwidth (Hz)
sampleRate = bandwidth;                                                 % Assume the sample rate is equal to the bandwidth

peakPower = 1.0;                                            % Peak power (W)
transmitter = phased.Transmitter('PeakPower', peakPower, 'Gain', 0);

noiseFigure = 3.0;                                                      % Noise figure (dB)
referenceTemperature = 290;                                             % Reference temperature (K)
receiver = phased.Receiver('SampleRate', sampleRate, 'NoiseFigure', noiseFigure,...
    'ReferenceTemperature', referenceTemperature, 'AddInputNoise', true,...
    'InputNoiseTemperature', referenceTemperature, 'Gain', 0);

Ntx = 8;                                                                % Number of Tx antenna elements
Nrx = 8;                                                                % Number of Rx antenna elements

element = phased.IsotropicAntennaElement('BackBaffled', true);
txArray = phased.ULA(Ntx, waveLength/2, 'Element', element);
rxArray = phased.ULA(Nrx, waveLength/2, 'Element', element);

%% ISAC Scenario
txPosition = [0; 0; 0];                                                 % Tx location
txOrientationAxis = eye(3);                                             % Tx array orientation

rxPosition = [80; 60; 0];                                               % Rx location
rxOrientationAxis = rotz(-90);                                          % Rx array orientation

maxPathLength = 300;                                                    % Maximum path lengths between Tx and Rx (m)
maxVelocity = 50;                                                       % Maximum relative velocity (m/s)

targetPositions = [60 70 90;                                            % Target positions (m)
                   -25 15 30;
                   0 0 0]; 

targetVelocities = [-15 20 0;                                           % Target velocities (m/s)
                    12 -10 25;
                    0 0 0];

% Platform to model target motion
targetMotion = phased.Platform('InitialPosition', targetPositions, 'Velocity', targetVelocities);

% The values of the reflection coefficients are chosen on random
targetReflectionCoefficients = randn(1, size(targetPositions, 2)) + 1i*randn(1, size(targetPositions, 2));

regionOfInterest = [0 120; -80 80];                                     % Bounds of the region of interest
numScatterers = 200;                                                    % Number of scatterers distributed within the region of interest
[scattererPositions, scattererReflectionCoefficients] = helperGenerateStaticScatterers(numScatterers, regionOfInterest);
channel = phased.ScatteringMIMOChannel('CarrierFrequency', carrierFrequency, 'TransmitArray', txArray,...
    'TransmitArrayPosition', txPosition, 'ReceiveArray', rxArray, 'ReceiveArrayPosition', rxPosition,...
    'TransmitArrayOrientationAxes',txOrientationAxis, 'ReceiveArrayOrientationAxes', rxOrientationAxis,...
    'SampleRate', sampleRate, 'SimulateDirectPath', true, 'ScattererSpecificationSource', 'Input Port');


helperVisualizeScatteringMIMOChannel(channel, scattererPositions, targetPositions, targetVelocities)
%title('Scattering MIMO Channel for Communication-Centric ISAC Scenario');

Nsub = 2048;                                                            % Number of subcarriers
subcarrierSpacing = bandwidth/Nsub;                                     % Separation between OFDM subcarriers
ofdmSymbolDuration = 1/subcarrierSpacing;                               % OFDM symbol duration

% Maximum Doppler shift based on the maximum relative velocity
maxDopplerShift = speed2dop(maxVelocity, waveLength);
fprintf("Subcarrier spacing is %.2f times larger than the maximum Doppler shift.\n", subcarrierSpacing/maxDopplerShift);

cyclicPrefixDuration = range2time(maxPathLength);                       % Duration of the cyclic prefix (CP)
cyclicPrefixLength = ceil(sampleRate*cyclicPrefixDuration);             % CP length in samples
cyclicPrefixDuration = cyclicPrefixLength/sampleRate;                   % Adjust duration of the CP to have an integer number of samples
Tofdm = ofdmSymbolDuration + cyclicPrefixDuration;                      % OFDM symbol duration with CP
ofdmSymbolLengthWithCP = Nsub + cyclicPrefixLength;                     % Number of samples in one OFDM symbol

% The first 9 and the last 8 subcarriers are used as guard bands
numGuardBandCarriers = [9; 8];

% Total number of subcarriers without guard bands
numActiveSubcarriers = Nsub-numGuardBandCarriers(1)-numGuardBandCarriers(2);

Mt = floor(1/(2*maxDopplerShift*Tofdm))

maxDelay = range2time(maxPathLength)/2;
Mf = floor(1/(2*subcarrierSpacing*maxDelay))

frameLength = Mt;

subframeALength = frameLength - Ntx;
subframeBLength = Ntx;

% Initial Channel Sounding
% Indices of the non-null preamble subcarriers at the first transmit antenna
idxs = [(numGuardBandCarriers(1)+1):Ntx:(Nsub/2-Ntx+1)...
    (Nsub/2+2):Ntx:(Nsub-numGuardBandCarriers(2)-Ntx+1)]';
numPreambleSubcarriers = numel(idxs);

% Shift subcarrier indices by one at each subsequent transmit antenna
preambleIdxs = zeros(numPreambleSubcarriers, 1, Ntx);
for i = 1:Ntx
    preambleIdxs(:, 1, i) = idxs + 1*(i-1);
end

% Use a known sequence as a preamble. The same values are transmitted by all of
% the transmit antennas.
preamble = mlseq(Nsub - 1);
preamble = preamble(1 : numPreambleSubcarriers);
preamble = repmat(preamble, 1, 1, Ntx);

preambleMod = comm.OFDMModulator("CyclicPrefixLength", cyclicPrefixLength,...
    "FFTLength", Nsub, "NumGuardBandCarriers", numGuardBandCarriers,...
    "NumSymbols", 1, "NumTransmitAntennas", Ntx,...
    "PilotCarrierIndices", preambleIdxs, "PilotInputPort", true);
preambleInfo = info(preambleMod);

preambleDemod = comm.OFDMDemodulator(preambleMod);
preambleDemod.NumReceiveAntennas = Nrx;

% When channel sounding is performed, almost all sub carries are used for
% the preamble. Null the remaining subcarriers.
preambleSignal = preambleMod(zeros(preambleInfo.DataInputSize), preamble);

% Transmit signal
txSignal = transmitter(preambleSignal);

% Apply scattering MIMO channel propagation effects
channelSignal = channel(txSignal, [scattererPositions targetPositions],...
    [zeros(size(scattererPositions)) targetVelocities],...
    [scattererReflectionCoefficients targetReflectionCoefficients]); 

% Add thermal noise at the receiver
rxSignal = receiver(channelSignal);

% Demodulate the received signal
[~, rxPreamblePilots] = preambleDemod(rxSignal);

% Estimate channel matrix
channelMatrix = helperInterpolateChannelMatrix(Nsub, numGuardBandCarriers, squeeze(preamble), squeeze(rxPreamblePilots), preambleIdxs);

% Compute precoding and combining weights
[Wp, Wc, ~, G] = diagbfweights(channelMatrix);

% Data Frame Transmission
% Subframe A contains only data
subframeAMod = comm.OFDMModulator("CyclicPrefixLength", cyclicPrefixLength,...
    "FFTLength", Nsub, "NumGuardBandCarriers", numGuardBandCarriers,...
    "NumTransmitAntennas", Ntx, "NumSymbols", subframeALength);
subframeAInfo = info(subframeAMod);

subframeADemod = comm.OFDMDemodulator(subframeAMod);
subframeADemod.NumReceiveAntennas = Nrx;

pilotIdxs = [(numGuardBandCarriers(1)+1):Mf:(Nsub/2) (Nsub/2+2):Mf:(Nsub-numGuardBandCarriers(2))]';
pilots = zeros(numel(pilotIdxs), Ntx, Ntx);
for itx = 1:Ntx
    s = mlseq(Nsub-1, itx);
    pilots(:, itx, itx) = s(1:numel(pilotIdxs));
end

subframeBMod = comm.OFDMModulator("CyclicPrefixLength", cyclicPrefixLength,...
    "FFTLength", Nsub, "NumGuardBandCarriers", numGuardBandCarriers,...
    "NumTransmitAntennas", Ntx, "NumSymbols", Ntx,...
    "PilotCarrierIndices", pilotIdxs, "PilotInputPort", true);
subframeBInfo = info(subframeBMod);

subframeBDemod = comm.OFDMDemodulator(subframeBMod);
subframeBDemod.NumReceiveAntennas = Nrx;

% Indices of data subcarriers in the subframe B
subframeBdataSubcarrierIdxs = setdiff(numGuardBandCarriers(1)+1:(Nsub-numGuardBandCarriers(2)), pilotIdxs);

Nframe = 24;                                                            % Total number of transmitted OFDM frames
fprintf("Velocity resolution: %.2f (m/s).\n", dop2speed(1/(Nframe*Tofdm*Mt), waveLength));

bitsPerSymbol = 6;                                                      % Bits per QAM symbol (and OFDM data subcarrier)
modOrder = 2^bitsPerSymbol;                                             % Modulation order

numDataStreams = 2;                                                     % Number of data streams

% Input data size for subframe A
subframeAInputSize = [subframeAInfo.DataInputSize(1) subframeAInfo.DataInputSize(2) numDataStreams];

% Input data size for subframe B
subframeBInputSize = [subframeBInfo.DataInputSize(1) subframeBInfo.DataInputSize(2) numDataStreams];

radarDataCube = zeros(numActiveSubcarriers, Nrx, Nframe);

for i = 1:Nframe
    % Generate binary payload for subframe A and modulate data using QAM
    subframeABin = randi([0,1], [subframeAInputSize(1) * bitsPerSymbol subframeAInputSize(2) numDataStreams]);
    subframeAQam = qammod(subframeABin, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);
    
    % Precode data subcarriers for subframe A
    subframeAQamPre = zeros(size(subframeAQam, 1), subframeALength, Ntx);
    for nsc = 1:numActiveSubcarriers
        subframeAQamPre(nsc, :, :) = squeeze(subframeAQam(nsc, :, :))*squeeze(Wp(nsc, 1:numDataStreams,:));
    end
    
    % Generate OFDM symbols for subframe A
    subframeA = subframeAMod(subframeAQamPre);

    % Generate binary payload for subframe B and modulate data using QAM
    subframeBBin = randi([0,1], [subframeBInputSize(1) * bitsPerSymbol subframeBInputSize(2) numDataStreams]);
    subframeBQam = qammod(subframeBBin, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);

    % Precode data subcarriers for subframe B
    subframeBQamPre = zeros(size(subframeBQam, 1), Ntx, Ntx);    
    for nsc = 1:numel(subframeBdataSubcarrierIdxs)
        idx = subframeBdataSubcarrierIdxs(nsc) - numGuardBandCarriers(1);
        subframeBQamPre(nsc, :, :) = squeeze(subframeBQam(nsc, :, :))*squeeze(Wp(idx, 1:numDataStreams,:));
    end

    % Generate OFDM symbols for subframe B
    subframeB = subframeBMod(subframeBQamPre, pilots);

    % Binary data transmitted in the ith frame
    txDataBin = cat(1, subframeABin(:), subframeBBin(:));
    
    % Reshape and combine subframes A and B to transmit the whole frame
    % one symbol at a time
    subframeA = reshape(subframeA, ofdmSymbolLengthWithCP, subframeALength, []);
    subframeB = reshape(subframeB, ofdmSymbolLengthWithCP, Ntx, []);
    ofdmSignal = [subframeA subframeB];

    % Preallocate space for the received signal
    rxSignal = zeros(size(ofdmSignal, 1), size(ofdmSignal, 2), Nrx);

    % Transmit one OFDM symbol at a time
    for s = 1:size(ofdmSignal, 2)
        % Update target positions
        [targetPositions, targetVelocities] = targetMotion(Tofdm);

        % Transmit signal
        txSignal = transmitter(squeeze(ofdmSignal(:, s, :)));        
        
        % Apply scattering MIMO channel propagation effects
        channelSignal = channel(txSignal, [scattererPositions targetPositions],...
            [zeros(size(scattererPositions)) targetVelocities],...
            [scattererReflectionCoefficients targetReflectionCoefficients]); 
        
        % Add thermal noise at the receiver
        rxSignal(:, s, :) = receiver(channelSignal);
    end

    % Separate the received signal into subframes A and B
    rxSubframeA = rxSignal(:, 1:subframeALength, :);
    rxSubframeA = reshape(rxSubframeA, [], Nrx);

    rxSubframeB = rxSignal(:, subframeALength+1:end, :);
    rxSubframeB = reshape(rxSubframeB, [], Nrx);

    % Demodulate subframe A and apply the combining weights
    rxSubframeAQam = subframeADemod(rxSubframeA);
    rxSubframeAQamComb = zeros(size(rxSubframeAQam, 1), size(rxSubframeAQam, 2), numDataStreams);

    for nsc = 1:numActiveSubcarriers
        rxSubframeAQamComb(nsc, :, :) = ((squeeze(rxSubframeAQam(nsc, :, :))*squeeze(Wc(nsc, :, 1:numDataStreams))))./sqrt(G(nsc,1:numDataStreams));
    end

    % Demodulate subframe B and apply the combining weights
    [rxSubframeBQam, rxPilots] = subframeBDemod(rxSubframeB);
    rxSubframeBQamComb = zeros(size(rxSubframeBQam, 1), size(rxSubframeBQam, 2), numDataStreams);

    for nsc = 1:numel(subframeBdataSubcarrierIdxs)
        idx = subframeBdataSubcarrierIdxs(nsc) - numGuardBandCarriers(1);
        rxSubframeBQamComb(nsc, :, :) = ((squeeze(rxSubframeBQam(nsc, :, :))*squeeze(Wc(idx, :, 1:numDataStreams))))./sqrt(G(idx, 1:numDataStreams));
    end

    % Demodulate the QAM data and compute the bit error rate for the ith
    % frame
    rxDataQam = cat(1, rxSubframeAQamComb(:), rxSubframeBQamComb(:));
    rxDataBin = qamdemod(rxDataQam, modOrder, 'OutputType', 'bit', 'UnitAveragePower', true);
    [~, ratio] = biterr(txDataBin, rxDataBin);
    fprintf("Frame %d bit error rate: %.4f\n", i, ratio);

    % Estimate channel matrix using pilots in the subframe B
    channelMatrix = helperInterpolateChannelMatrix(Nsub, numGuardBandCarriers, pilots, rxPilots, pilotIdxs);
    
    % Compute precoding and combining weights for the next frame
    [Wp, Wc, ~, G] = diagbfweights(channelMatrix);

    % Store the radar data
    radarDataCube(:, :, i) = squeeze(sum(channelMatrix, 2));    
end

refconst = qammod(0:modOrder-1, modOrder, 'UnitAveragePower', true);
constellationDiagram = comm.ConstellationDiagram('NumInputPorts', 1, ...
    'ReferenceConstellation', refconst, 'ChannelNames', {'Received QAM Symbols'});

constellationDiagram(rxDataQam);

%% Radar Data Processing
% Perform FFT over the slow-time dimension and zero out the DC component to
% remove static scatterers
Y = fft(radarDataCube, [], 3);
Y(:, :, 1) = 0;
y = ifft(Y, Nframe, 3);

% Plot position heat map
phm = helperPositionHeatmap('ReceiveArray', rxArray, 'ReceiveArrayOrientationAxis', rxOrientationAxis, 'ReceiveArrayPosition', rxPosition, ...
    'SampleRate', sampleRate, 'CarrierFrequency', carrierFrequency, 'Bandwidth', bandwidth, 'OFDMSymbolDuration', ofdmSymbolDuration, ...
    'TransmitArrayOrientationAxis', txOrientationAxis, 'TransmitArrayPosition', txPosition, 'TargetPositions', targetPositions, 'ROI', [0 120; -80 80]);

figure(1);
phm.plot(y)
title('Moving Scatterers');

rangeDopplerResponse = phased.RangeDopplerResponse('RangeMethod', 'FFT', ...
    'SampleRate', sampleRate, 'SweepSlope', bandwidth/ofdmSymbolDuration,...
    'PRFSource', 'Property', 'PRF', 1/(Tofdm*frameLength), 'ReferenceRangeCentered', false);

[rdr, r, doppler] = rangeDopplerResponse(conj(y));
doppler = doppler * (-1);   % -1 to account for conj in the range-do response

% Combine signals from all receive antennas using non-coherent integration
rdr_integ = squeeze(sum(abs(rdr), 2));

figure(2);
imagesc(doppler, r*2, rdr_integ);
ax = gca;
set(ax, 'YDir', 'normal') ;

colorbar;
xlabel('Frequency (Hz)');
ylabel('Sum Range (m)');
title('Range-Doppler Response');
grid on;
ylim([0 maxPathLength]);
hold on;

% Bistatic Doppler
bistaticDoppler = helperBistaticDopplerShift(txPosition, rxPosition, targetPositions, targetVelocities, carrierFrequency);

% Sum range
sumRange = vecnorm(targetPositions - rxPosition) + vecnorm(targetPositions - txPosition);

plot(bistaticDoppler, sumRange, 'o', 'LineWidth', 1, 'MarkerSize', 28, 'Color', '#D95319',...
    'MarkerFaceColor', 'none', 'DisplayName', 'Targets of interest')
legend



















