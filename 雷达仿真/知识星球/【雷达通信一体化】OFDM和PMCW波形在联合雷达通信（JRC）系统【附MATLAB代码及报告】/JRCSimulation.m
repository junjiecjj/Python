%% PART 1
rng('default');

basicParams = struct( ...
    'carrierFreq', 24e9, ...
    'bandwidth', 100e6, ...
    'peakPower', 0.01 , ...
    'TxAntennaGain', 20, ...
    'RxAntennaGain', 20, ...
    'noiseFigure', 2.5,...
    'refTemperature',290,...
    'maxRange',100,...
    'maxRelVelocity', 50 ...
);


JRCMotion = struct( ...
       'position', [0; 0; 0], ...
       'velocity', [0; 0; 0] ...
    );

JRCPlatform =  phased.Platform('InitialPosition', JRCMotion.position , 'Velocity', JRCMotion.velocity);
                    
target = struct( ...
        'positions' , [85 60 45; 15 -5 0; 0 0 0],...
        'velocities' ,  [10 -20 -18; 0 0 0; 0 0 0],...
        'radarCrossSection' ,  [1.8 5.3 3.8]...
    );

targetMotion = phased.Platform('InitialPosition', target.positions, 'Velocity', target.velocities);

radarTarget = phased.RadarTarget('Model', 'Swerling1', 'MeanRCS', target.radarCrossSection, 'OperatingFrequency', basicParams.carrierFreq);

userCoordinates = [50; 50; 0];                                  

JRCGraph(JRCMotion.position, target.positions, userCoordinates, target.velocities);

%% JRC System Using a PMCW Waveform

PRBS = struct();
    PRBS.a = 8;
    PRBS.sequence = helperMLS(PRBS.a);
    PRBS.chipNumber = numel(PRBS.sequence);
    PRBS.chipDuration = 1/basicParams.bandwidth;  
    PRBS.modPeriod = PRBS.chipNumber *  PRBS.chipDuration;

CommData = struct();
    CommData.bits = 256;
    CommData.binary =  randi([0, 1], [CommData.bits 1]);
    CommData.pskSymbols = pskmod(CommData.binary, 2);

TransmitWaveform = struct();
    TransmitWaveform.xWave = [PRBS.sequence * ones(1, CommData.bits); PRBS.sequence * CommData.pskSymbols.'];
    TransmitWaveform.period = 2*PRBS.modPeriod; 

 % Radar Signal Simulation and Processing

 RadarSim = struct();
    RadarSim.sampleFreq = basicParams.bandwidth;
    RadarSim.transmitter = phased.Transmitter('Gain', basicParams.TxAntennaGain, 'PeakPower', basicParams.peakPower);
    RadarSim.antenna = phased.IsotropicAntennaElement;
    RadarSim.radiator = phased.Radiator('Sensor', RadarSim.antenna, 'OperatingFrequency', basicParams.carrierFreq);
    RadarSim.collector = phased.Collector('Sensor', RadarSim.antenna, 'OperatingFrequency', basicParams.carrierFreq);
    RadarSim.reciever = phased.ReceiverPreamp('SampleRate', RadarSim.sampleFreq, 'Gain', basicParams.RxAntennaGain, 'NoiseFigure', basicParams.noiseFigure, 'ReferenceTemperature', basicParams.refTemperature);
    RadarSim.channel = phased.FreeSpace('SampleRate', RadarSim.sampleFreq, 'TwoWayPropagation', true, 'OperatingFrequency', basicParams.carrierFreq);
    RadarSim.recievedYWave =  zeros(size(TransmitWaveform.xWave));

 


for loop = 1:CommData.bits
    
    
    [JRCMotion.position, JRCMotion.velocity] = JRCPlatform(TransmitWaveform.period);
    [target.positions, target.velocities] = targetMotion(TransmitWaveform.period);
    
  
    % Calculate the target angles as seen from the transmit array
    [targetRange, targetAngle] = rangeangle(target.positions, JRCMotion.position);

    TransmittedSignal = struct();
        TransmittedSignal.signal = RadarSim.transmitter(TransmitWaveform.xWave(:, loop));
        TransmittedSignal.radiatedSignal = RadarSim.radiator(TransmittedSignal.signal, targetAngle);
        TransmittedSignal.channelEffect = RadarSim.channel(TransmittedSignal.radiatedSignal, JRCMotion.position, target.positions, JRCMotion.velocity, target.velocities);
        TransmittedSignal.reflectedSignal = radarTarget(TransmittedSignal.channelEffect, false);
        TransmittedSignal.receivedSignal = RadarSim.collector(TransmittedSignal.reflectedSignal, targetAngle);

 
    % Add thermal noise at the receiver
    RadarSim.recievedYWave(:, loop) = RadarSim.reciever(TransmittedSignal.receivedSignal);
end

FilteredSignal = struct();
    FilteredSignal.recivedYWave = RadarSim.recievedYWave(PRBS.chipNumber+1:end, :) .* (CommData.pskSymbols.');
    FilteredSignal.recivedYWave = FilteredSignal.recivedYWave + RadarSim.recievedYWave(1:PRBS.chipNumber, :);
    FilteredSignal.fftYWave = fft(FilteredSignal.recivedYWave);
    FilteredSignal.fftSequence = fft(PRBS.sequence);
    FilteredSignal.FDomSignal = FilteredSignal.fftYWave .* conj(FilteredSignal.fftSequence);
 

% Range-Doppler response object computes DFT in the slow-time domain and
% then IDFT in the fast-time domain to obtain the range-Doppler map
RangeDopplerResponse = phased.RangeDopplerResponse( ...
    'RangeMethod', 'FFT', ...
    'SampleRate', RadarSim.sampleFreq, ...
    'SweepSlope', -basicParams.bandwidth / PRBS.modPeriod, ...
    'DopplerOutput', 'Speed', ...
    'OperatingFrequency', basicParams.carrierFreq, ...
    'PRFSource', 'Property', ...
    'PRF', 1 / TransmitWaveform.period, ...
    'ReferenceRangeCentered', false ...
);


[response, rangeGrid, dopplerGrid] = step(RangeDopplerResponse, FilteredSignal.FDomSignal);

figure;
imagesc(dopplerGrid, rangeGrid, 10*log10(abs(response)));
set(gca, 'YDir', 'normal'); % Ensure the y-axis isn't flipped
xlabel('Relative Velocity (m/s)');
ylabel('Range (m)');
colorbar;
title('Range-Doppler Response');
xlim([-basicParams.maxRelVelocity basicParams.maxRelVelocity]);
ylim([0 basicParams.maxRange]);





%Communication Signal Simulation and proccessing

CommSim = struct();
    CommSim.DownlinkLOS = vecnorm(JRCMotion.position - userCoordinates);
    CommSim.LOS = CommSim.DownlinkLOS/physconst('LightSpeed');
    CommSim.Delay = CommSim.LOS * [1 1.6 1.1 1.45] - CommSim.LOS;
    CommSim.Gain = [0 -1 2 -1.5];
    CommSim.dopplerFreq = 2 * speed2dop(basicParams.maxRelVelocity, freq2wavelen(basicParams.carrierFreq));
    CommSim.channel = comm.RicianChannel( ...
        'PathGainsOutputPort', true, ...
        'DirectPathDopplerShift', 0, ...
        'MaximumDopplerShift', CommSim.dopplerFreq / 2, ...
        'PathDelays', CommSim.Delay, ...
        'AveragePathGains', CommSim.Gain, ...
        'SampleRate', RadarSim.sampleFreq ...
    );
    CommSim.SigNoiseRatio = 40;
    CommSim.TXSignal = reshape(awgn(CommSim.channel(TransmitWaveform.xWave(:)), CommSim.SigNoiseRatio, 'measured'), 2*PRBS.chipNumber, []);
    CommSim.freqResponse = fft(CommSim.TXSignal(1:PRBS.chipNumber, :))./FilteredSignal.fftSequence;
    CommSim.freqDomRecievedSig = fft(CommSim.TXSignal(PRBS.chipNumber + 1:end, :));
    CommSim.timeDomSig = ifft((CommSim.freqDomRecievedSig .* conj(FilteredSignal.fftSequence))./CommSim.freqResponse);


[~, index] = max(abs(CommSim.timeDomSig), [], 'linear');
    CommSim.recievedData = pskdemod(CommSim.timeDomSig(index).'./ PRBS.chipNumber, 2);
    
diagConst = comm.ConstellationDiagram(...
    'NumInputPorts', 1, ...
    'ReferenceConstellation', pskmod([0 1], 2), ...
    'ChannelNames', {'Received PSK Symbols'});

diagConst(CommSim.timeDomSig(index).'./ PRBS.chipNumber(:));

%% OFDM

OFDM = struct();
    OFDM.subcarrier = 1024;
    OFDM.sep = basicParams.bandwidth/OFDM.subcarrier;
    OFDM.duration = 1/OFDM.sep;
    OFDM.CPDuration = range2time(basicParams.maxRange);
    OFDM.CPLength = ceil(RadarSim.sampleFreq*OFDM.CPDuration);
    OFDM.CPDuration = OFDM.CPLength/RadarSim.sampleFreq;
    OFDM.symDur = OFDM.duration + OFDM.CPDuration;
    OFDM.symNum = OFDM.subcarrier + OFDM.CPLength;
    OFDM.dataSubcarrier = OFDM.subcarrier-length([1:9 (OFDM.subcarrier/2+1) (OFDM.subcarrier-8:OFDM.subcarrier)]');

empIndex = [1:9 (OFDM.subcarrier/2+1) (OFDM.subcarrier-8:OFDM.subcarrier)]';

TransmitData = struct();
    TransmitData.bits = 6;
    TransmitData.order = 2^TransmitData.bits;
    TransmitData.numofSym = 128 ;
    
    CommData.binary = randi([0,1], [OFDM.dataSubcarrier*TransmitData.bits TransmitData.numofSym]);

    TransmitData.modulated = reshape(ofdmmod(qammod(CommData.binary, ...
                                     TransmitData.order, 'InputType', 'bit', 'UnitAveragePower', true), ...
                                     OFDM.subcarrier, OFDM.CPLength, empIndex), ...
                                     OFDM.symNum, ...
                                     TransmitData.numofSym);

recievedRadarSig = zeros(size(TransmitData.modulated/max(sqrt(abs(TransmitData.modulated).^2), [], 'all')));

% Reset the platform objects representing the JRC and the targets before
% running the simulation
reset(JRCPlatform);
reset(targetMotion);


for loop = 1:TransmitData.numofSym
    [JRCMotion.position, JRCMotion.velocity] = JRCPlatform(OFDM.symDur);

    [target.positions, target.velocities] = targetMotion(OFDM.symDur);

    [targetRange, targetAngle] = rangeangle(target.positions, JRCMotion.position);

    TransmittedSignal.signal = RadarSim.transmitter(TransmitData.modulated(:, loop));

    TransmittedSignal.radiatedSignal = RadarSim.radiator(TransmittedSignal.signal, targetAngle);

    TransmittedSignal.channelEffect = RadarSim.channel(TransmittedSignal.radiatedSignal, JRCMotion.position, target.positions, JRCMotion.velocity, target.velocities);

    TransmittedSignal.reflectedSignal = radarTarget(TransmittedSignal.channelEffect, false);

    TransmittedSignal.receivedSignal = RadarSim.collector(TransmittedSignal.reflectedSignal, targetAngle);
    
    recievedRadarSig(:, loop) = RadarSim.reciever(TransmittedSignal.receivedSignal);
end

recievedRadarSig1 = reshape(recievedRadarSig, OFDM.symNum*TransmitData.numofSym, 1);

% Demodulate the received OFDM signal (compute DFT)
TransmitData.demodulated = ofdmdemod(recievedRadarSig1, OFDM.subcarrier, OFDM.CPLength, OFDM.CPLength, empIndex);

TransmitData.filtered = TransmitData.demodulated./qammod(CommData.binary, TransmitData.order, 'InputType', 'bit', 'UnitAveragePower', true);

% Range-Doppler response object computes DFT in the slow-time domain and
% then IDFT in the fast-time domain to obtain the range-Doppler map
RangeDopplerResponse2 = phased.RangeDopplerResponse('RangeMethod', 'FFT', 'SampleRate', RadarSim.sampleFreq, 'SweepSlope', -basicParams.bandwidth/OFDM.symDur,...
    'DopplerOutput', 'Speed', 'OperatingFrequency', basicParams.carrierFreq, 'PRFSource', 'Property', 'PRF', 1/OFDM.symDur, ...
    'ReferenceRangeCentered', false); 

figure;
plotResponse(RangeDopplerResponse2, TransmitData.filtered, 'Unit', 'db');
xlim([-basicParams.maxRelVelocity basicParams.maxRelVelocity]);
ylim([0 basicParams.maxRange]);


ChannelResponse = struct();
    

[propagation, PG] = CommSim.channel(ofdmmod(qammod(CommData.binary, ...               %pg path gain
                                     TransmitData.order, 'InputType', 'bit', 'UnitAveragePower', true), ...
                                     OFDM.subcarrier, OFDM.CPLength, empIndex));

ChannelResponse.propagation = awgn(propagation, CommSim.SigNoiseRatio, "measured");


ChannelResponse.PF = info(CommSim.channel).ChannelFilterCoefficients;   %pf path filters
ChannelResponse.OF = info(CommSim.channel).ChannelFilterDelay;      %of offset
ChannelResponse.ofdm = ofdmChannelResponse(PG, ChannelResponse.PF, OFDM.subcarrier, OFDM.CPLength, setdiff(1:OFDM.subcarrier, empIndex), ChannelResponse.OF);

ChannelResponse.QAMEqualized = ofdmEqualize(ofdmdemod([ChannelResponse.propagation(ChannelResponse.OF+1:end,:); ...
                    zeros(ChannelResponse.OF, 1)], OFDM.subcarrier, OFDM.CPLength, OFDM.CPLength/2, empIndex), ...
                    ChannelResponse.ofdm(:), 'Algorithm', 'zf');


CommSim.recievedData = qamdemod(ChannelResponse.QAMEqualized, TransmitData.order, 'OutputType', 'bit', 'UnitAveragePower', true);


diagConst = comm.ConstellationDiagram('NumInputPorts', 1, ...
    'ReferenceConstellation', qammod(0:TransmitData.order-1, TransmitData.order, 'UnitAveragePower', true), 'ChannelNames', {'Received QAM Symbols'});


diagConst(ChannelResponse.QAMEqualized(1:OFDM.dataSubcarrier*10).');

