%% Subfunción: estimación inicial del canal y pesos de precodificación
function config = initialChannelEstimation(config)
    % This function performs the initial channel estimation and precoding for the ISAC system.  

    % Extract from config
    systemParams = config.systemParams;
    elements     = config.elements;
    scenario     = config.scenario;
    ofdm         = config.ofdm;

    transmitter = elements.transmitter;
    receiver    = elements.receiver;
    channel     = elements.channel;

    Ntx  = systemParams.Ntx;    % Number of transmit systemParams
    Nrx  = systemParams.Nrx;    % Number of receive systemParams
    Nsub = ofdm.Nsub;       % Total number of subcarriers

    % Indices of the non-null preamble subcarriers at the first transmit antenna
    idxs = [(ofdm.numGuardBandCarriers(1)+1):Ntx:(Nsub/2 - Ntx + 1), ...
            (Nsub/2 + 2):Ntx:(Nsub - ofdm.numGuardBandCarriers(2) - Ntx + 1)]';
    numPreambleSubcarriers = numel(idxs);

    % Shift subcarrier indices by one at each subsequent transmit antenna
    preambleIdxs = zeros(numPreambleSubcarriers, 1, Ntx);
    for i = 1:Ntx
        preambleIdxs(:, 1, i) = idxs + (i - 1);
    end

    % Use a known sequence as a preamble.
    % The same values are transmitted by all of the transmit systemParams.
    preamble = mlseq(Nsub - 1);
    preamble = preamble(1 : numPreambleSubcarriers);
    preamble = repmat(preamble, 1, 1, Ntx);

    % Create OFDM modulator for preamble
    preambleMod = comm.OFDMModulator( ...
        'CyclicPrefixLength', ofdm.cyclicPrefixLength, ...
        'FFTLength', Nsub, ...
        'NumGuardBandCarriers', ofdm.numGuardBandCarriers, ...
        'NumSymbols', 1, ...
        'NumTransmitAntennas', Ntx, ...
        'PilotCarrierIndices', preambleIdxs, ...
        'PilotInputPort', true);

    % Create OFDM demodulator for preamble
    preambleDemod = comm.OFDMDemodulator(preambleMod);
    preambleDemod.NumReceiveAntennas = Nrx;

    % When channel sounding is performed, almost all subcarriers are used for
    % the preamble. Null the remaining subcarriers.
    preambleSignal = preambleMod(zeros(info(preambleMod).DataInputSize), preamble);

    % Transmit signal
    txSignal = transmitter(preambleSignal);

    % Apply scattering MIMO channel propagation effects
    channelSignal = channel(txSignal, ...
        [scenario.scatterPos, scenario.targetPositions], ...
        [zeros(size(scenario.scatterPos)), scenario.targetVelocities], ...
        [scenario.scatterRC, scenario.targetRC]);

    % Add thermal noise at the receiver
    rxSignal = receiver(channelSignal);

    % Demodulate the received signal
    [~, rxPreamblePilots] = preambleDemod(rxSignal);

    % Estimate channel matrix
    channelMatrix = helperInterpolateChannelMatrix(Nsub, ofdm.numGuardBandCarriers, ...
        squeeze(preamble), squeeze(rxPreamblePilots), preambleIdxs);

    % Compute precoding and combining weights
    [Wp, Wc, ~, G] = diagbfweights(channelMatrix);

    precoding = struct();
    precoding.Wp = Wp;
    precoding.Wc = Wc;
    precoding.G  = G;

    % Append precoding and channel matrix to config
    config.channelMatrix = channelMatrix;
    config.precoding = precoding;
end