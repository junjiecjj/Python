%% Subfunción: configuración del escenario ISAC
function config = configureScenario(config)
    % This function configures the scenario for the ISAC system, including
    % transmitter and receiver positions, target motion, and static scatterers.

    % Extract system parameters and arrays from config
    systemParams = config.systemParams;
    arrays = config.elements.arrays;

    % Initialize scenario structure
    scenario = struct();

    % Transmitter and receiver location and orientation
    scenario.txPos = [0; 0; 0];              % Transmitter position
    scenario.rxPos = [80; 60; 0];            % Receiver position
    scenario.txAxis = eye(3);                % Transmitter orientation (identity matrix)
    scenario.rxAxis = rotz(-90);             % Receiver orientation (rotation around Z-axis by -90 degrees)

    % Target and scatterer parameters
    scenario.maxRange = 300;                 % Maximum path lengths between Tx and Rx (m)
    scenario.maxVelocity = 50;               % Maximum target velocity (m/s)

    % Target positions and velocities
    scenario.targetPositions = [60 70 90; -25 15 30; 0 0 0];          % Target positions (m)
    scenario.targetVelocities = [-15 20 0; 12 -10 25; 0 0 0];         % Target velocities (m/s)

    % Platform to model target motion
    scenario.targetMotion = phased.Platform( ...
        'InitialPosition', scenario.targetPositions, ...
        'Velocity', scenario.targetVelocities);

    % The values of the reflection coefficients are chosen randomly
    scenario.targetRC = randn(1, size(scenario.targetPositions, 2)) + ...
                        1i * randn(1, size(scenario.targetPositions, 2));

    % Generate static scatterers
    scenario.regionOfInterest = [0 120; -80 80];   % Bounds of the region of interest
    scenario.numScatterers = 200;                  % Number of scatterers distributed within the region of interest
    [scenario.scatterPos, scenario.scatterRC] = ...
        helperGenerateStaticScatterers(scenario.numScatterers, scenario.regionOfInterest);

    % Create MIMO scattering channel
    channel = phased.ScatteringMIMOChannel( ...
        'CarrierFrequency', systemParams.carrierFrequency, ...
        'TransmitArray', arrays.tx, ...
        'ReceiveArray', arrays.rx, ...
        'TransmitArrayPosition', scenario.txPos, ...
        'ReceiveArrayPosition', scenario.rxPos, ...
        'TransmitArrayOrientationAxes', scenario.txAxis, ...
        'ReceiveArrayOrientationAxes', scenario.rxAxis, ...
        'SampleRate', systemParams.sampleRate, ...
        'SimulateDirectPath', true, ...
        'ScattererSpecificationSource', 'Input port');

    % Visualization and saving logic
    if isfield(config, 'options') && config.options.SHOW_IMAGES
        helperVisualizeScatteringMIMOChannel(channel, scenario.scatterPos, ...
                                             scenario.targetPositions, ...
                                             scenario.targetVelocities);
        title('Scattering MIMO Channel for Communication-Centric ISAC Scenario');

        % Save the figure if enabled
        if config.options.SAVE_IMAGES
            if isfield(config.options, 'figSaveFolder') && isfield(config.options, 'figPrefix')
                filename = fullfile(config.options.figSaveFolder, [config.options.figPrefix, 'scenario.png']);
                saveas(gcf, filename);
            else
                warning('Figure not saved: figSaveFolder or figPrefix missing in config.options.');
            end
        end
    end

    % Add scenario to the config structure
    config.scenario = scenario;
    config.elements.channel = channel;
end
