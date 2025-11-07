%% Subfunción: configuración OFDM (Signaling Scheme)
function config = configureOFDM(config)
    % This function configures the OFDM parameters for the ISAC system.

    % Extract necessary fields from config
    systemParams = config.systemParams;
    scenario     = config.scenario;

    % Initialize OFDM configuration
    ofdm = struct();

    ofdm.Nsub = systemParams.NumSubcarriers;  % Number of subcarriers
    ofdm.subcarrierSpacing = systemParams.bandwidth / ofdm.Nsub; % Separation between OFDM subcarriers (Hz)
    ofdm.ofdmSymbolDuration = 1 / ofdm.subcarrierSpacing; % OFDM symbol duration
    ofdm.Nframe = systemParams.Nframe; % Total number of transmitted OFDM frames

    % Maximum Doppler shift based on the maximum relative velocity
    ofdm.maxDopplerShift = speed2dop(scenario.maxVelocity, systemParams.waveLength);
    fprintf("Subcarrier spacing is %.2f times larger than the maximum Doppler shift.\n", ...
        ofdm.subcarrierSpacing / ofdm.maxDopplerShift);

    % Calculate cyclic prefix duration and length
    cyclicPrefixDuration = range2time(scenario.maxRange);                          % Duration of the cyclic prefix (CP)
    ofdm.cyclicPrefixLength = ceil(systemParams.sampleRate * cyclicPrefixDuration); % CP length in samples
    ofdm.cyclicPrefixDuration = ofdm.cyclicPrefixLength / systemParams.sampleRate;  % Adjust duration of the CP to have an integer number of samples

    % Calculate OFDM symbol duration and length with cyclic prefix
    ofdm.Tofdm = ofdm.ofdmSymbolDuration + ofdm.cyclicPrefixDuration;       % OFDM symbol duration with CP
    ofdm.ofdmSymbolLengthWithCP = ofdm.Nsub + ofdm.cyclicPrefixLength;      % Number of samples in one OFDM symbol

    % The first 9 and the last 8 subcarriers are used as guard bands
    ofdm.numGuardBandCarriers = [9; 8];
    % Total number of subcarriers without guard bands
    ofdm.numActiveSubcarriers = ofdm.Nsub - sum(ofdm.numGuardBandCarriers);

    % Calculate Doppler and range sampling parameters
    ofdm.Mt = floor(1 / (2 * ofdm.maxDopplerShift * ofdm.Tofdm));        % Doppler sampling parameter
    ofdm.maxDelay = range2time(scenario.maxRange) / 2;                   % Maximum delay
    ofdm.Mf = floor(1 / (2 * ofdm.subcarrierSpacing * ofdm.maxDelay));   % Range sampling parameter

    % Calculate the frame and subframe lengths
    ofdm.frameLength = ofdm.Mt;                        % Frame length
    ofdm.subframeALength = ofdm.frameLength - systemParams.Ntx; % Subframe A length
    ofdm.subframeBLength = systemParams.Ntx;               % Subframe B length

    % Other OFDM parameters
    ofdm.numDataStreams = systemParams.numDataStreams; % Number of data streams
    ofdm.bitsPerSymbol = systemParams.bitsPerSymbol;   % Bits per QAM symbol (and OFDM data subcarrier)
    ofdm.modOrder = 2^ofdm.bitsPerSymbol;              % Modulation order

    % Append OFDM configuration to config
    config.ofdm = ofdm;
end
