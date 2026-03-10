% https://ww2.mathworks.cn/help/phased/ug/integrated-sensing-and-communication-1-radar-centric-approach-using-pmcw-waveform.html


clc;
clear all;
close all;
addpath('./functions');

% Set the random number generator for reproducibility 
rng('default');

carrierFrequency = 77e9;                                                % Carrier frequency (Hz)
waveLength = freq2wavelen(carrierFrequency);                            % Wavelength
bandwidth = 150e6;                                                      % Bandwidth (Hz)
sampleRate = bandwidth;                                                 % Assume the sample rate is equal to the bandwidth

peakPower = 1;                                                          % Peak power (W)
transmitter = phased.Transmitter('PeakPower', peakPower, 'Gain', 0);

noiseFigure = 3.0;                                                      % Noise figure (dB)
referenceTemperature = 1;                                               % Reference temperature (K)
radarReceiver = phased.Receiver('SampleRate', sampleRate, 'NoiseFigure', noiseFigure,...
     'ReferenceTemperature', referenceTemperature, 'AddInputNoise', true,...
     'InputNoiseTemperature', referenceTemperature, 'Gain', 0);    


Ntx = 4;                                                                % Number of transmit antenna elements
element = phased.IsotropicAntennaElement('BackBaffled', true);
txArray = phased.ULA(Ntx, waveLength/2, 'Element', element);

Nrx = 4;                                                                % Number of receive antenna elements at the radar receiver
rxArray = phased.ULA(Nrx,Ntx*waveLength/2, 'Element', element);

downlinkNoiseFigure = 3.3;                                              % Noise figure at the downlink receiver (dB)
downlinkReferenceTemperature = 290;                                     % Reference temperature at the downlink receiver (dB)
downlinkReceiver = phased.Receiver('SampleRate', sampleRate, 'NoiseFigure', downlinkNoiseFigure,...
    'ReferenceTemperature', downlinkReferenceTemperature, 'AddInputNoise', true,...
    'InputNoiseTemperature', downlinkReferenceTemperature, 'Gain', 0);

downlinkAntenna = phased.IsotropicAntennaElement;

%% ISAC Scenario
radarPosition = [0; 0; 0];                                              % Location of the transmitter (m)
radarOrientationAxis = eye(3);                                          % Tx array orientation

userPosition = [80; 60; 0];                                             % Location of the receiver (m)

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

[targetAzimuth, ~, targetRange] = cart2sph(targetPositions(1, :), targetPositions(2, :), targetPositions(3, :));
targetAzimuth = rad2deg(targetAzimuth);
targetRadialVelocity = sum(targetVelocities.*(targetPositions./vecnorm(targetPositions)));

for i = 1:size(targetPositions, 2)
    fprintf("Target %d: range=%.2f (m), azimuth=%.2f (deg), radial velocity = %.2f\n", i,...
        targetRange(i), targetAzimuth(i), targetRadialVelocity(i));
end

regionOfInterest = [0 120; -80 80];                                     % Bounds of the region of interest
numScatterers = 200;                                                    % Number of scatterers distributed within the region of interest
[scattererPositions, scattererReflectionCoefficients] = helperGenerateStaticScatterers(numScatterers, regionOfInterest);

radarChannel = phased.ScatteringMIMOChannel('CarrierFrequency', carrierFrequency, 'TransmitArray', txArray,...
    'TransmitArrayPosition', radarPosition, 'ReceiveArray', rxArray, 'ReceiveArrayPosition', radarPosition,...
    'TransmitArrayOrientationAxes', radarOrientationAxis, 'ReceiveArrayOrientationAxes', radarOrientationAxis,...
    'SampleRate', sampleRate, 'SimulateDirectPath', false, 'ScattererSpecificationSource', 'Input Port');

commChannel = phased.ScatteringMIMOChannel('CarrierFrequency', carrierFrequency, 'TransmitArray', txArray,...
    'TransmitArrayPosition', radarPosition, 'ReceiveArray', downlinkAntenna, 'ReceiveArrayPosition', userPosition,...
    'TransmitArrayOrientationAxes', radarOrientationAxis, 'SampleRate', sampleRate,...
    'SimulateDirectPath', true, 'ScattererSpecificationSource', 'Input Port');

helperVisualizeScatteringMIMOChannel(radarChannel, scattererPositions, targetPositions, targetVelocities)
plot(userPosition(2), userPosition(1), 'pentagram', 'DisplayName', 'Downlink user Rx', 'MarkerSize', 16,...
    'Color', '#A2142F', 'MarkerFaceColor', '#A2142F')
title('Scattering MIMO Channel for Radar-Centric ISAC Scenario');


%%  Signaling Scheme
Nchip = 255;                                                            % Length of the PRBS
prbs = mlseq(Nchip);                                                    % Maximum length sequence

chipWidth = 1/bandwidth;                                                % Chip duration
modulationPeriod = Nchip * chipWidth;                                   % Modulation period

fprintf("Maximum unambiguous range: %.2f (m).\n", time2range(modulationPeriod));
H = hadamard(Ntx);
outterCodedPRBS = kron(H, prbs);

blockDuration = modulationPeriod * Ntx;
blockLength = Nchip * Ntx;

Nb = 256;                                                               % Total number of transmitted blocks

M = 32;                                                                 % Perform sounding every Mth block 
numSoundBlocks = numel(1:M:Nb);                                         % Total number of sounding blocks
fprintf("Channel matrix update period: %.2f (ms).\n", blockDuration*M*1e3);

txDataBin = ones(Nb-numSoundBlocks, Ntx);
txDataBpsk = ones(Nb, Ntx);

radarRxSignal = zeros(blockLength, Nrx, Nb);
commRxSignal = zeros(blockLength, Nb + 1);

it = 0;
for i = 1:Nb
    if mod(i - 1, M) == 0   % Communication channel sounding
        % Transmit the outer codded PRBS without any data added to it
        pmcwSignal = outterCodedPRBS;
    else                    % Data transmission
        % Generate binary payload and peform BPSK modulation
        it = it + 1;
        txDataBin(it, :) = randi([0 1], [1 Ntx]);
        txDataBpsk(i, :) = pskmod(txDataBin(it, :), 2); 

        % Modulate the outer coded PRBS with the BPSK data payload
        pmcwSignal = outterCodedPRBS .* txDataBpsk(i, :);    
    end

    % Transmit signal
    txSignal = transmitter(pmcwSignal);
    
    % Update target positions
    [targetPositions, targetVelocities] = targetMotion(blockDuration);

    % Simulate signal propagation through the radar channel - from
    % transmitter to the scatterers and back to the radar receiver
    radarChannelSignal = radarChannel(txSignal, [scattererPositions targetPositions],...
        [zeros(size(scattererPositions)) targetVelocities],...
        [scattererReflectionCoefficients targetReflectionCoefficients]); 

    % Simulate signal propagation through the comm channel - from
    % transmitter to the scatterers and to the downlink user
    commChannelSignal = commChannel(txSignal, [scattererPositions targetPositions],...
        [zeros(size(scattererPositions)) targetVelocities],...
        [scattererReflectionCoefficients targetReflectionCoefficients]); 
       
    % Add thermal noise at the receiver
    radarRxSignal(:, :, i) = radarReceiver(radarChannelSignal);
    commRxSignal(:, i) = downlinkReceiver(commChannelSignal);
end

commChannelSignal = commChannel(zeros(size(txSignal)), [scattererPositions targetPositions],...
    [zeros(size(scattererPositions)) targetVelocities],...
    [scattererReflectionCoefficients targetReflectionCoefficients]); 

commRxSignal(:, end) = downlinkReceiver(commChannelSignal);

radarRxSignalDecoded = helperDecodeMIMOPMCW(radarRxSignal, H);
for i = 1:Nb
    radarRxSignalDecoded(:, :, :, i) = radarRxSignalDecoded(:, :, :, i) .* txDataBpsk(i, :); 
end

radarDataCube = reshape(radarRxSignalDecoded, Nchip, Ntx*Nrx, Nb);

prbsDFT = fft(prbs);
Z = fft(radarDataCube) .* conj(prbsDFT);

Y = fft(Z, [], 3);
Y(:, :, 1) = 0;
y = ifft(Y, Nb, 3);

virtualArray = helperVirtualArray(txArray, rxArray);
rangeAngleResponse = phased.RangeAngleResponse('SensorArray', virtualArray, 'RangeMethod', 'FFT', 'SampleRate', sampleRate,...
    'SweepSlope', bandwidth/modulationPeriod, 'OperatingFrequency', carrierFrequency, 'ReferenceRangeCentered', false);

[rar, r, theta] = rangeAngleResponse(conj(y));
theta = theta * (-1);   % -1 to account for conj in the range-angle response

rar_integ = sum(abs(rar), 3);

figure(1);
imagesc(theta, r, rar_integ);
ax = gca;
set(ax, 'YDir', 'normal') ;

colorbar;
xlabel('Azimuth Angle (deg)');
ylabel('Range (m)');
title('Range-Angle Response');
grid on;
ylim(regionOfInterest(1, :));
hold on;
plot(targetAzimuth, targetRange, 'o', 'LineWidth', 1, 'MarkerSize', 32, 'Color', '#D95319',...
    'MarkerFaceColor', 'none', 'DisplayName', 'Targets of interest');
legend;

rangeDopplerResponse = phased.RangeDopplerResponse('RangeMethod', 'FFT', 'DopplerOutput', 'Speed',...
    'OperatingFrequency', carrierFrequency, 'SampleRate', sampleRate, 'SweepSlope', bandwidth/modulationPeriod,...
    'PRFSource', 'Property', 'PRF', 1/blockDuration, 'ReferenceRangeCentered', false);

[rdr, r, doppler] = rangeDopplerResponse(conj(y));
doppler = doppler * (-1);   % -1 to account for conj in the range-do response

figure(2);
rdr_integ = squeeze(sum(abs(rdr), 2));

figure;
imagesc(doppler, r, rdr_integ);
ax = gca;
set(ax, 'YDir', 'normal') ;

colorbar;
xlabel('Speed (m/s)');
ylabel('Range (m)');
title('Range-Doppler Response');
grid on;
xlim([-50 50]);
ylim(regionOfInterest(1, :));
hold on;
plot(targetRadialVelocity, targetRange, 'o', 'LineWidth', 1, 'MarkerSize', 32, 'Color', '#D95319',...
    'MarkerFaceColor', 'none', 'DisplayName', 'Targets of interest');
legend;


commRxBuffer = [commRxSignal(:, 1:Nb); commRxSignal(:, 2:(Nb+1))];
d = helperCommChannelDelay(commRxBuffer(1:Nchip, :), prbs);

idx = sub2ind(size(commRxBuffer), d + (0:blockLength-1).', repmat(1:Nb, blockLength, 1));
commRxBuffer = commRxBuffer(idx);

commRxSignalDecoded = helperDecodeMIMOPMCW(commRxBuffer, H);

rxDataBpsk = zeros([Nb-numSoundBlocks Ntx]);
ir = 0;

for i = 1:Nb
    if mod(i - 1, M) == 0  % Estimate channel matrix if unmodulated PRBS was transmitted
        channelMatrix = fft(commRxSignalDecoded(:, :, i))./prbsDFT;  
    else
        % Perform channel equalization
        commRxSignalEqualized = (fft(commRxSignalDecoded(:, :, i)) .* conj(prbsDFT))./channelMatrix;
        y = ifft(commRxSignalEqualized);
    
        % Demodulate the signal to obtain the BPSK symbols.
        [~, idx] = max(abs(y), [], 'linear');
    
        ir = ir + 1;
        rxDataBpsk(ir, :) = y(idx)/Nchip;
    end
end

refconst = pskmod([0 1], 2);
constellationDiagram = comm.ConstellationDiagram('NumInputPorts', 1, ...
    'ReferenceConstellation', refconst, 'ChannelNames', {'Received QAM Symbols'});

constellationDiagram(rxDataBpsk(:));

rxDataBin = pskdemod(rxDataBpsk(:), 2);
[numErr,ratio] = biterr(txDataBin(:), rxDataBin(:));


