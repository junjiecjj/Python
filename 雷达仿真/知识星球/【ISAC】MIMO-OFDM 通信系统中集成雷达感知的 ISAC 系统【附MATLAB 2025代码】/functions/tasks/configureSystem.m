%% Subfunción: configuración del sistema
function config = configureSystem(params)
    % This function configures the system parameters, communication elements
    % (transmitter, receiver, arrays), and visualization options based on the
    % provided parameters. Returns a single structured config object.

        % ==== DEFAULT PARAMETERS ====
    defaultParams = struct( ...
        'CarrierFreq_GHz',        6, ...      % GHz
        'Bandwidth_MHz',          100, ...     % MHz
        'NumSubcarriers',         2048, ...
        'numDataStreams',         2, ...
        'Nframe',                 24, ...
        'bitsPerSymbol',          6, ...       
        'peakPower_W',            1, ...
        'noiseFigure_dB',         3, ...
        'referenceTemperature_K', 290, ...
        'TxAntennas',             8, ...
        'RxAntennas',             8, ...
        'figSaveFolder',          'default', ...
        'figPrefix',              '' ...
    );

    % ==== MERGE DEFAULTS WITH USER PARAMS ====
    paramFields = fieldnames(defaultParams);
    for i = 1:numel(paramFields)
        field = paramFields{i};
        if ~isfield(params, field)
            params.(field) = defaultParams.(field);
        end
    end

    % ==== SYSTEM PARAMETERS ====
    systemParams = struct();

    % Initial parameters
    systemParams.carrierFrequency     = params.CarrierFreq_GHz * 1e9;                 % Carrier frequency (Hz)
    systemParams.waveLength           = freq2wavelen(systemParams.carrierFrequency);  % Wavelength (m)
    systemParams.bandwidth            = params.Bandwidth_MHz * 1e6;                   % Bandwidth (Hz)
    systemParams.sampleRate           = systemParams.bandwidth;                       % Sample rate (Hz)

    systemParams.NumSubcarriers       = params.NumSubcarriers;        % Number of subcarriers
    systemParams.numDataStreams       = params.numDataStreams;        % Number of data streams
    systemParams.Nframe               = params.Nframe;                % Total number of transmitted OFDM frames
    systemParams.bitsPerSymbol        = params.bitsPerSymbol;         % Bits per QAM symbol
    systemParams.modOrder             = 2^systemParams.bitsPerSymbol; % Modulation order
    systemParams.peakPower            = params.peakPower_W;           % Peak power (W)
    systemParams.noiseFigure          = params.noiseFigure_dB;        % Noise figure (dB)
    systemParams.referenceTemperature = params.referenceTemperature_K;% Reference temperature (K)
    systemParams.Ntx                  = params.TxAntennas;                                % Number of transmit antennas
    systemParams.Nrx                  = params.RxAntennas;                                % Number of receive antennas

    % ==== COMMUNICATION ELEMENTS ====
    % Transmitter
    transmitter = phased.Transmitter( ...
        'PeakPower', systemParams.peakPower, ...
        'Gain', 0);

    % Receiver
    receiver = phased.Receiver( ...
        'SampleRate', systemParams.sampleRate, ...
        'NoiseFigure', systemParams.noiseFigure, ...
        'ReferenceTemperature', systemParams.referenceTemperature, ...
        'AddInputNoise', true, ...
        'InputNoiseTemperature', systemParams.referenceTemperature, ...
        'Gain', 0);

    % Antenna arrays
    element = phased.IsotropicAntennaElement('BackBaffled', true);  % Isotropic antenna element with back baffling

    arrays = struct();
    arrays.tx = phased.ULA(systemParams.Ntx, systemParams.waveLength / 2, 'Element', element); % Tx array
    arrays.rx = phased.ULA(systemParams.Nrx, systemParams.waveLength / 2, 'Element', element); % Rx array

    % Combine all physical/simulated blocks
    elements = struct();
    elements.transmitter = transmitter;
    elements.receiver    = receiver;
    elements.arrays      = arrays;
    % (Later: elements.channel = ...)

    % ==== VISUALIZATION AND SAVING OPTIONS ====
    options = struct();
    options.SHOW_IMAGES = isfield(params, 'SHOW_IMAGES') && params.SHOW_IMAGES;
    options.SAVE_IMAGES = isfield(params, 'SAVE_IMAGES') && params.SAVE_IMAGES;

    options.figSaveFolder = fullfile('results', params.figSaveFolder);
    options.figPrefix = params.figPrefix;

    % Create the folder if it does not exist
    if ~exist(options.figSaveFolder, 'dir')
        mkdir(options.figSaveFolder);
    end

    % ==== FINAL CONFIG STRUCT ====
    config = struct();
    config.systemParams = systemParams;
    config.elements     = elements;
    config.options      = options;
end
