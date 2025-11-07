function run_isac_simulation(params)
    % Set the random number generator for reproducibility
    rng('default');

    %% 1. System Parameters
    config = configureSystem(params);

    %% 2. Escenario ISAC
    config = configureScenario(config);

    %% 3. Configuración OFDM y estimación de canal (initial channel sounding)
    config = configureOFDM(config);
    config = initialChannelEstimation(config);

    %% 4. Transmisión de tramas OFDM y evaluación de BER
    radarDataCube = transmitDataFrames(config);
    
    %% 5. Procesado radar y métricas de sensado.
    processRadarData(config, radarDataCube);
end