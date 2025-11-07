%% Subfunción: procesado radar y métricas de sensado
function processRadarData(config, radarDataCube)
    % This function processes the radar data and computes various sensing metrics.

    % Extract required fields from config
    systemParams = config.systemParams;
    scenario     = config.scenario;
    arrays       = config.elements.arrays;
    ofdm         = config.ofdm;
    options      = config.options;

    % Perform FFT over the slow-time dimension and zero out the DC component to
    % remove static scatterers
    Y = fft(radarDataCube, [], 3);
    Y(:, :, 1) = 0;
    y = ifft(Y, systemParams.Nframe, 3);

    % Plot position heat map
    phm = helperPositionHeatmap( ...
        'ReceiveArray', arrays.rx, ...
        'ReceiveArrayOrientationAxis', scenario.rxAxis, ...
        'ReceiveArrayPosition', scenario.rxPos, ...
        'SampleRate', systemParams.sampleRate, ...
        'CarrierFrequency', systemParams.carrierFrequency, ...
        'Bandwidth', systemParams.bandwidth, ...
        'OFDMSymbolDuration', ofdm.ofdmSymbolDuration, ...
        'TransmitArrayOrientationAxis', scenario.txAxis, ...
        'TransmitArrayPosition', scenario.txPos, ...
        'TargetPositions', scenario.targetPositions, ...
        'ROI', scenario.regionOfInterest);

    if options.SHOW_IMAGES
        figure;
        phm.plot(y);
        title('Moving Scatterers');

        if options.SAVE_IMAGES
            filename = fullfile(options.figSaveFolder, [options.figPrefix, 'heatmap.png']);
            saveas(gcf, filename);
        end
    end

    % Configure range-Doppler response
    rangeDopplerResponse = phased.RangeDopplerResponse( ...
        'RangeMethod', 'FFT', ...
        'SampleRate', systemParams.sampleRate, ...
        'SweepSlope', systemParams.bandwidth / ofdm.ofdmSymbolDuration, ...
        'PRFSource', 'Property', ...
        'PRF', 1 / (ofdm.Tofdm * ofdm.frameLength), ...
        'ReferenceRangeCentered', false);

    [rdr, r, doppler] = rangeDopplerResponse(conj(y));
    doppler = doppler * (-1); % -1 to account for conj in the range-doppler response

    % Combine signals from all receive antennas using non-coherent integration
    rdr_integ = squeeze(sum(abs(rdr), 2));

    if options.SHOW_IMAGES
        figure;
        imagesc(doppler, r * 2, rdr_integ);
        ax = gca;
        set(ax, 'YDir', 'normal');
        colorbar;
        xlabel('Frequency (Hz)');
        ylabel('Sum Range (m)');
        title('Range-Doppler Response');
        grid on;
        ylim([0 scenario.maxRange]);
        hold on;

        % Calculate bistatic Doppler and sum range for target visualization
        bistaticDoppler = helperBistaticDopplerShift( ...
            scenario.txPos, scenario.rxPos, ...
            scenario.targetPositions, scenario.targetVelocities, ...
            systemParams.carrierFrequency);

        % Sum range
        sumRange = vecnorm(scenario.targetPositions - scenario.rxPos) + ...
                   vecnorm(scenario.targetPositions - scenario.txPos);

        plot(bistaticDoppler, sumRange, 'o', ...
            'LineWidth', 1, ...
            'MarkerSize', 28, ...
            'Color', '#D95319', ...
            'MarkerFaceColor', 'none', ...
            'DisplayName', 'Targets of interest');
        legend;

        if options.SAVE_IMAGES
            filename = fullfile(options.figSaveFolder, [options.figPrefix, 'range_doppler.png']);
            saveas(gcf, filename);
        end
    end
end
