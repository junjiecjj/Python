%% Subfunción: transmisión de tramas OFDM y recolección de datos radar
function radarDataCube = transmitDataFrames(config)
    % This function transmits OFDM frames and evaluates the Bit Error Rate (BER) 

    % Extract from config
    systemParams = config.systemParams;
    scenario     = config.scenario;
    ofdm         = config.ofdm;
    precoding    = config.precoding;
    transmitter  = config.elements.transmitter;
    channel      = config.elements.channel;
    receiver     = config.elements.receiver;
    options      = config.options;

    % Subframe A
    [subframeAMod, subframeADemod, subframeAInfo] = createModDemod(true, ofdm, systemParams);

    % Subframe B
    % Define pilot subcarrier indices for subframe B
    [ofdm.pilotIdxs, ofdm.pilots] = generatePilots(ofdm, systemParams.Ntx);
    [subframeBMod, subframeBDemod, subframeBInfo] = createModDemod(false, ofdm, systemParams);

    % Indices of data subcarriers in the subframe B
    ofdm.subframeBdataSubcarrierIdxs = setdiff( ...
        ofdm.numGuardBandCarriers(1)+1 : (ofdm.Nsub - ofdm.numGuardBandCarriers(2)), ...
        ofdm.pilotIdxs);

    velocityResolution = dop2speed(1 / (ofdm.Nframe * ofdm.Tofdm * ofdm.Mt), systemParams.waveLength);
    fprintf("Velocity resolution: %.2f (m/s).\n", velocityResolution);

    % Input data size for subframe A and B
    subAInputSize = [subframeAInfo.DataInputSize(1), subframeAInfo.DataInputSize(2), ofdm.numDataStreams];
    subBInputSize = [subframeBInfo.DataInputSize(1), subframeBInfo.DataInputSize(2), ofdm.numDataStreams];

    % Initialize radar data cube
    radarDataCube = zeros(ofdm.numActiveSubcarriers, systemParams.Nrx, ofdm.Nframe);

    % Initialize BER tracking
    berRatios = zeros(ofdm.Nframe, 1);

    % Simulate formation, transmission, and reception of an OFDM frame one at a time.
    for i = 1:ofdm.Nframe
        % Generate binary payload for subframes A and B and modulate data using QAM
        [subframeABin, subA] = generateSubframe(true, subframeAMod, subAInputSize, systemParams, ofdm, precoding, systemParams);
        [subframeBBin, subB] = generateSubframe(false, subframeBMod, subBInputSize, systemParams, ofdm, precoding, systemParams);

        % Binary data transmitted in the ith frame
        txDataBin = cat(1, subframeABin(:), subframeBBin(:));

        % Reshape and combine subframes A and B to transmit the whole frame one symbol at a time
        subA = reshape(subA, ofdm.ofdmSymbolLengthWithCP, ofdm.subframeALength, []);
        subB = reshape(subB, ofdm.ofdmSymbolLengthWithCP, systemParams.Ntx, []);
        ofdmSignal = [subA subB];

        % Preallocate space for the received signal
        rxSignal = zeros(size(ofdmSignal,1), size(ofdmSignal,2), systemParams.Nrx);

        % Transmit one OFDM symbol at a time
        for s = 1:size(ofdmSignal,2)
            % Update target positions
            [scenario.targetPositions, scenario.targetVelocities] = scenario.targetMotion(ofdm.Tofdm);

            % Transmit signal
            tx = transmitter(squeeze(ofdmSignal(:,s,:)));

            % Apply scattering MIMO channel propagation effects
            chanOut = channel(tx,...
                [scenario.scatterPos scenario.targetPositions], ...
                [zeros(size(scenario.scatterPos)) scenario.targetVelocities], ...
                [scenario.scatterRC scenario.targetRC]);

            % Add thermal noise at the receiver
            rxSignal(:,s,:) = receiver(chanOut);
        end

        % Separate the received signal into subframes A and B
        rxSubframeA = rxSignal(:, 1:ofdm.subframeALength, :);
        rxSubframeA = reshape(rxSubframeA, [], systemParams.Nrx);

        rxSubframeB = rxSignal(:, ofdm.subframeALength+1:end, :);
        rxSubframeB = reshape(rxSubframeB, [], systemParams.Nrx);

        % Demodulate subframe A and B and apply the combining weights
        [rxSubframeAQamComb, ~] = demodulateAndApplyWeights(true, subframeADemod, rxSubframeA, ofdm, precoding, systemParams);
        [rxSubframeBQamComb, rxPilots] = demodulateAndApplyWeights(false, subframeBDemod, rxSubframeB, ofdm, precoding, systemParams);

        % Demodulate the QAM data and compute the bit error rate for the ith frame
        rxDataQam = cat(1, rxSubframeAQamComb(:), rxSubframeBQamComb(:));
        rxDataBin = qamdemod(rxDataQam, ofdm.modOrder, 'OutputType', 'bit', 'UnitAveragePower', true);
        [~, ratio] = biterr(txDataBin, rxDataBin);
        fprintf("Frame %d bit error rate: %.4f\n", i, ratio);
        berRatios(i) = ratio; % Store BER for this frame

        % Channel estimation and precoding update
        channelMatrix = helperInterpolateChannelMatrix(ofdm.Nsub, ofdm.numGuardBandCarriers, ofdm.pilots, rxPilots, ofdm.pilotIdxs);
        [Wp, Wc, ~, G] = diagbfweights(channelMatrix);
        precoding.Wp = Wp;
        precoding.Wc = Wc;
        precoding.G = G;

        radarDataCube(:,:,i) = squeeze(sum(channelMatrix, 2));
    end

    % Calculate and save mean BER
    meanBER = mean(berRatios);
    fprintf("Mean BER across all frames: %.6f\n", meanBER);
    
    % Create filename with timestamp and configuration info
    filename = fullfile(options.figSaveFolder, [options.figPrefix, 'BER_results.txt']);

    % Save BER results to text file
    fileID = fopen(filename, 'w');
    fprintf(fileID, 'ISAC Simulation BER Results\n');
    fprintf(fileID, '===========================\n');
    fprintf(fileID, 'Per-frame BER:\n');
    for i = 1:ofdm.Nframe
        fprintf(fileID, 'Frame %2d: %.6f\n', i, berRatios(i));
    end
    fprintf(fileID, '\nMean BER: %.6f\n', meanBER);
    fclose(fileID);
    
    fprintf("BER results saved to: %s\n", filename);


    if config.options.SHOW_IMAGES
        figure;
        scatterplot(rxDataQam);
        title('Received QAM Symbols');
        grid on;
    
        if config.options.SAVE_IMAGES
            filename = fullfile(options.figSaveFolder, [options.figPrefix, 'constellation.png']);
            saveas(gcf, filename);  % Saves current figure
        end
    end
end