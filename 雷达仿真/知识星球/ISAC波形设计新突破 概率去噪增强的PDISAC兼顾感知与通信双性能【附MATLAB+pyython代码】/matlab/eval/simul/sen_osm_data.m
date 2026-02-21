siteviewer( ...
    Buildings="/data/polytechnique.osm", ...
    Basemap="satellite");


fc = 24e9; % Carrier frequency (Hz)
radar_position = [45.501361,-73.614219, 155]; % lat, lon, alt
radar_ant_size = [1 1];                    % number of rows and columns in rectangular array (base station)
radar_array_orientation = [0 0].';       % azimuth (0 deg is East, 90 deg is North) and elevation (positive points upwards) in deg
radar_vel = [0; 0; 0];
radar_location = [0; 0; 0];


ue_position = [45.501520,-73.613956, 147]; % lat, lon, alt
ue_ant_size = [1 1];                    % number of rows and columns in rectangular array (base station)
ue_array_orientation = [0 0].';       % azimuth (0 deg is East, 90 deg is North) and elevation (positive points upwards) in deg
ue_reflections_order = 1;                 % number of reflections for ray tracing analysis (0 for LOS)
[x, y, z] = latlon2local(ue_position(1), ue_position(2), ue_position(3), radar_position);
ue_location = [x; y; z];  % Local position of the target
ue_rcs = 1.3;


target_position = [45.501014,-73.614031, 147];  % lat, lon, alt
target_reflections_order = 0;                 % number of reflections for ray tracing analysis (0 for LOS)
[x, y, z] = latlon2local(target_position(1), target_position(2), target_position(3), radar_position);
target_location = [x; y; z];  % Local position of the target
target_rcs = 4.7;


radarSite = txsite("Name","Radar", ...
    "Latitude",radar_position(1),"Longitude",radar_position(2),...
    "AntennaAngle",radar_array_orientation(1:2),...
    "TransmitterFrequency",fc);

ueSite = rxsite("Name","UE", ...
    "Latitude",ue_position(1),"Longitude",ue_position(2),...
    "AntennaAngle",ue_array_orientation(1:2));


targetSite = rxsite("Name","Target", ...
    "Latitude",target_position(1),"Longitude",target_position(2));

% Compute rays at ue
pm = propagationModel("raytracing","Method","sbr","MaxNumReflections",ue_reflections_order,"MaxNumDiffractions",0);
uerays = raytrace(radarSite,ueSite,pm,"Type","pathloss");

pm = propagationModel("raytracing","Method","sbr","MaxNumReflections",target_reflections_order,"MaxNumDiffractions",0);
targetrays = raytrace(radarSite,targetSite,pm,"Type","pathloss");


% Preview the topology
show(radarSite);
show(ueSite);
show(targetSite);
plot(uerays{1});
plot(targetrays{1});

time_steps = 0;


% Speeds: start → end (linear change)
v_tar_range = [30, 30];   % target speeds m/s
v_ue_range  = [6, 6];  % UE speeds m/s

% Simulation positions and velocities
[radar_location, target_locations, ue_locations, tar_vels, ue_vels] = ...
    update_positions_linear(radar_location, target_location, ue_location, time_steps, v_tar_range, v_ue_range);

% Compute distance and speed
[tar_distances, ue_distances, tar_speeds, ue_speeds] = ...
    compute_distances_and_speeds(radar_location, target_locations, ue_locations, tar_vels, ue_vels);

% rng("default");
% Signal Settings
fc = 24e9; % Carrier frequency (Hz)
Gtx = 50; % Tx antenna gain (dB)
Grx = 50; % Radar Rx antenna gain (dB)
Grx_ue = 30; % UE Rx antenna gain (dB)
NF = 2.9; % Noise figure (dB)
Tref = 290; % Reference temperature (K)
Rmax = 200; % Maximum range of interest
vrelmax = 60; % Maximum relative velocity
c = physconst('LightSpeed');
lambda = c / fc;



num_of_bits = 256;
num_of_chips = 255;

P_prbs = load("data/p_freq_255.mat").data;
% P_prbs = load("data/p_freq_2550.mat").data;
p_prbs = ifft(P_prbs);

% p_prbs = 2*randi([0,1], num_of_chips, 1) - 1;
% P_prbs = fft(p_prbs);


T_d = 5.1e-6;
T_chip = T_d / (2 * num_of_chips);
T_prbs = T_d / 2;
T_D = T_d * num_of_bits;
T_frame = T_D / 2;
fs = 1 / T_chip;
B = fs;

rdr = phased.RangeDopplerResponse(...
    'RangeMethod', 'FFT', ...
    'SampleRate', fs, ...
    'SweepSlope', -B/T_prbs, ...
    'DopplerOutput', 'Speed', ...
    'OperatingFrequency', fc, ...
    'PRFSource', 'Property', ...
    'PRF', 1/T_d, ...
'ReferenceRangeCentered', false);


% Setting hyperparameter for sensing
num_of_trials = 100;
SNRs_Tx = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50];


NF_lin = 10^(NF / 10);
noise_power = physconst('Boltzmann') * Tref * NF_lin * B;

source_dir = "data/sensing_osm/";

for idx_SNR = 1:length(SNRs_Tx)
    SNR_tx = SNRs_Tx(idx_SNR);
    SNR_tx
    Pt = noise_power * 10^(SNR_tx/10);
    for idx_time = 1:length(time_steps)
        time_step = time_steps(idx_time);

        radar_motion = phased.Platform('InitialPosition', radar_location, 'Velocity', radar_vel);
        target_motion = phased.Platform('InitialPosition', target_locations(:, idx_time), 'Velocity', tar_vels(:, idx_time));
        ue_motion = phased.Platform('InitialPosition', ue_locations(:, idx_time), 'Velocity', ue_vels(:, idx_time));


        target_distance = tar_distances(:, idx_time);
        ue_distance = ue_distances(:, idx_time);
        target_speed = tar_speeds(:, idx_time);
        ue_speed = ue_speeds(:, idx_time);

        

        % Radar components setup for target (Radar -> Target -> Radar)
        transmitter_tar = phased.Transmitter('Gain', Gtx, 'PeakPower', Pt);
        ant_tar = phased.IsotropicAntennaElement;
        radiator_tar = phased.Radiator('Sensor', ant_tar, 'OperatingFrequency', fc);
        collector_tar = phased.Collector('Sensor', ant_tar, 'OperatingFrequency', fc);
        receiver_tar = phased.ReceiverPreamp('SampleRate', fs, 'Gain', Grx, 'NoiseFigure', NF, 'ReferenceTemperature', Tref);
        receiver_tar_no_noise = phased.ReceiverPreamp('Gain', Grx, 'NoiseMethod', 'Noise power', 'NoisePower', 0); 
        radar_to_target_channel = phased.FreeSpace('SampleRate', fs, 'TwoWayPropagation', false, 'OperatingFrequency', fc);
        target_to_radar_channel = phased.FreeSpace('SampleRate', fs, 'TwoWayPropagation', false, 'OperatingFrequency', fc);
        target_tar = phased.RadarTarget('Model', 'Swerling1', 'MeanRCS', target_rcs, 'OperatingFrequency', fc);

        % Radar components setup for ue (Radar -> UE -> Radar)
        transmitter_ue = phased.Transmitter('Gain', Gtx, 'PeakPower', Pt);
        ant_ue = phased.IsotropicAntennaElement;
        radiator_ue = phased.Radiator('Sensor', ant_ue, 'OperatingFrequency', fc);
        collector_ue = phased.Collector('Sensor', ant_ue, 'OperatingFrequency', fc);
        receiver_ue = phased.ReceiverPreamp('SampleRate', fs, 'Gain', Grx, 'NoiseFigure', NF, 'ReferenceTemperature', Tref);

        path_delays = [uerays{1}.PropagationDelay]-min([uerays{1}.PropagationDelay]);
        average_path_gains = -[uerays{1}.PathLoss];
        path_delays = path_delays * 0.01;
        average_path_gains_lin = 10.^(average_path_gains/10);
        K = average_path_gains_lin(1) / sum(average_path_gains_lin(2:end)); % Linear K
        fdmax = 2 * speed2dop(ue_speed, freq2wavelen(fc));
        commChannel = comm.RicianChannel( ...
            'PathGainsOutputPort', true, ...
            'DirectPathDopplerShift', 0, ...     % LOS path static
            'MaximumDopplerShift', fdmax, ...        % scattered paths static
            'PathDelays', path_delays, ...
            'AveragePathGains', average_path_gains, ...
            'SampleRate', fs, ...
            'NormalizePathGains', true, ...
            'KFactor', K, ...
            'RandomStream', 'mt19937ar with seed', ...
            'Seed', 12345 ...
        ); 

        radar_to_ue_channel = phased.FreeSpace('SampleRate', fs, 'TwoWayPropagation', false, 'OperatingFrequency', fc);
        ue_to_radar_channel = phased.FreeSpace('SampleRate', fs, 'TwoWayPropagation', false, 'OperatingFrequency', fc);
        target_ue = phased.RadarTarget('Model', 'Swerling1', 'MeanRCS', ue_rcs, 'OperatingFrequency', fc);


        total_errors = 0;
        normalied_power = 0;

        Gtx_lin = 10^(Gtx / 10); % Tx gain (linear)
        Grx_lin = 10^(Grx / 10); % Rx gain (linear)
        alpha_tar =  sqrt((Pt * Gtx_lin * Grx_lin * lambda^2 * target_rcs) / ((4*pi)^3 * target_distance^4));
        P_sensing = alpha_tar^2;

        % Give trial for each Noise Figures
        for idx_trial = 1:num_of_trials
            % Initialize input
            bits = randi([0, 1], [num_of_bits, 1]);
            s_pbsk = pskmod(bits, 2);

            M_prbs = p_prbs * ones(1, num_of_bits);
            M_pmcw = p_prbs * s_pbsk.';
            D_isac = [M_prbs; M_pmcw];


            % Initialize output
            y_D_isac = zeros(size(D_isac));
            y_D_isac_no_noise = zeros(size(D_isac));

            reset(radar_motion);
            reset(target_motion);
            reset(ue_motion);

            % Radar processing
            for m = 1:num_of_bits
                % Update radar position
                [cur_l_radar, cur_v_radar] = radar_motion(T_d);

                % ##################
                % # Target sensing #
                % ##################

                % Radar -> Target
                [cur_l_tar, cur_v_tar] = target_motion(T_d);
                [cur_r_tar, cur_a_tar] = rangeangle(cur_l_tar, cur_l_radar);
                tar_sig = transmitter_tar(D_isac(:, m));
                tar_radsig = radiator_tar(tar_sig, cur_a_tar);
                tar_forward_output = radar_to_target_channel(tar_radsig, cur_l_radar, cur_l_tar, cur_v_radar, cur_v_tar);

                % Target -> Radar
                tar_tgtsig = target_tar(tar_forward_output, false);
                tar_channel = target_to_radar_channel(tar_tgtsig, cur_l_tar, cur_l_radar, cur_v_tar, cur_v_radar);
                tar_rxsig = collector_tar(tar_channel, cur_a_tar);
                y_tar = receiver_tar(tar_rxsig);
                y_tar_no_noise = receiver_tar_no_noise(tar_rxsig);

                % ##################
                % # UE sensing #
                % ##################

                % Radar -> UE
                [cur_l_ue, cur_v_ue] = ue_motion(T_d);
                [cur_r_ue, cur_a_ue] = rangeangle(cur_l_ue, cur_l_radar);
                ue_sig = transmitter_ue(D_isac(:, m));
                ue_radsig = radiator_ue(ue_sig, cur_a_ue);

                ue_forward_output = radar_to_ue_channel(ue_radsig, cur_l_radar, cur_l_ue, cur_v_radar, cur_v_ue);
                [ue_forward_output, h_rician] = commChannel(ue_forward_output);

                % UE -> Radar
                ue_tgtsig = target_ue(ue_forward_output, false);
                ue_channel = ue_to_radar_channel(ue_tgtsig, cur_l_ue, cur_l_radar, cur_v_ue, cur_v_radar);
                ue_rxsig = collector_ue(ue_channel, cur_a_ue);
                y_ue = receiver_ue(ue_rxsig);
                y_ue_no_noise = receiver_tar_no_noise(tar_rxsig);


                % Combine target and UE signals
                y_D_isac(:, m) = y_tar + y_ue;
                % y_D_isac(:, m) = y_tar;

                y_D_isac_no_noise(:, m) = y_tar_no_noise + y_ue_no_noise;
                % y_D_isac_no_noise(:, m) = y_tar_no_noise;

            end
            
            y_PRBS_waveform_no_noise =  y_D_isac_no_noise(1:num_of_chips, :);
            y_PRBS_waveform =  y_D_isac(1:num_of_chips, :);
            % y_PRBS_waveform_no_noise = generate_signal_model_parametric(target_distance, target_speed, alpha_tar, p_prbs, T_chip, T_d, fc, c, num_of_chips, num_of_bits);
            % y_PRBS_waveform = add_complex_noise(y_PRBS_waveform_no_noise, noise_power);


            Y_PRBS_waveform = fft(y_PRBS_waveform);
            Z_PRBS_waveform = Y_PRBS_waveform .* conj(P_prbs);
            z_PRBS_waveform = ifft(Z_PRBS_waveform);

            Y_PRBS_waveform_no_noise = fft(y_PRBS_waveform_no_noise);
            Z_PRBS_waveform_no_noise = Y_PRBS_waveform_no_noise .* conj(P_prbs);
            z_PRBS_waveform_no_noise = ifft(Z_PRBS_waveform_no_noise);
            

            % [hat_r_tars, hat_v_tars] = est_rv_mle_y(y_PRBS_waveform, p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            % [hat_r_tars, hat_v_tars] = est_rv_seq_z(z_PRBS_waveform, T_prbs, 1, fs, lambda);
            % figure;
            % plotResponse(rdr, Z_PRBS_waveform, 'Unit', 'db');
            % xlim([-vrelmax vrelmax]);
            % ylim([0 Rmax]);
            % hold on;
            % scatter(hat_v_tars, hat_r_tars, 100, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
            % scatter([target_speed, ue_speed], [target_distance, ue_distance], 100, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
            % hold off;


            if idx_trial < 3
                waveform_types = {'Z_PRBS_waveform', 'Y_PRBS_waveform'};
                waveform_data = {Z_PRBS_waveform, Y_PRBS_waveform};


                % Loop through each waveform type to generate and save plots
                for i = 1:length(waveform_types)
                    % Create directory for waveform type and SNR
                    output_dir = fullfile(source_dir, "images", waveform_types{i}, num2str(time_step), num2str(SNR_tx));
                    if ~exist(output_dir, 'dir')
                        mkdir(output_dir);
                    end

                    % Create figure (no display)
                    figure('visible', 'off');

                    % Plot the waveform
                    plotResponse(rdr, waveform_data{i}, 'Unit', 'db');
                    xlim([-vrelmax vrelmax]);
                    ylim([0 Rmax]);

                    % Define filename and save as PNG
                    filename = fullfile(output_dir, sprintf('%s_%d.png', waveform_types{i}, idx_trial));
                    saveas(gcf, filename); % Save as PNG

                    % Close the figure to free memory
                    close(gcf);
                end
            end

            % ###############################
            % Save data in frequency domain #
            % ###############################
            waveform_types = {'Y_PRBS_waveform_no_noise', 'Y_PRBS_waveform'};
            waveform_data = {Y_PRBS_waveform_no_noise, Y_PRBS_waveform};


            % waveform_data = normalize_waveforms(waveform_data, 0, 1);


            % Loop through each waveform type to save data
            for i = 1:length(waveform_types)
                % Create directory for waveform type and SNR
                output_dir = fullfile(source_dir, "mats", "freq", waveform_types{i}, num2str(time_step), num2str(SNR_tx));
                if ~exist(output_dir, 'dir')
                    mkdir(output_dir);
                end

                % Save waveform data as .mat file
                data = waveform_data{i};
                filename = fullfile(output_dir, sprintf('%d.mat', idx_trial));
                save(filename, 'data');
            end

        end
    
    end

end


function S_model = generate_signal_model_parametric(r, v, A, p_prbs, T_chip, T_sym, fc, c, N_fast, N_slow)
    % Generate parametric signal model following:
    % y[n,m] = A * p[n-n0] * exp(j*(2*pi*fd*m*T_sym - phi0))
    %
    % where:
    %   n0 = round(2*R / (c*T_chip)) - delay in chips
    %   fd = 2*v*fc/c - Doppler frequency (Hz)
    %   phi0 = 4*pi*fc*R/c - carrier phase shift (radians)

    % Step 1: Compute delay in chips (round-trip)
    tau = 2 * r / c; % Round-trip delay (seconds)
    n0 = round(tau / T_chip); % Integer delay in chips

    % Step 2: Shift PRBS code by delay
    % p[n - n0] means we shift p to the right by n0 samples
    % In MATLAB: circshift with negative means shift left, positive means shift right
    % We want p[n] -> p[n-n0], which is shifting the sequence to the right
    n0_wrapped = mod(n0, N_fast); % Wrap around for circular shift
    p_shifted = circshift(p_prbs, n0_wrapped); % Shift right by n0

    % Step 3: Compute Doppler frequency
    fd = -2 * v * fc / c; % Hz

    % Step 4: Compute carrier phase shift
    phi0 = 4 * pi * fc * r / c; % radians

    % Step 5: Generate phase modulation for slow-time
    % Phase varies as: 2*pi*fd*m*T_sym for m = 0, 1, 2, ..., N_slow-1
    m_indices = 0:(N_slow-1); % Slow-time indices
    phase_doppler = 2 * pi * fd * m_indices * T_sym; % 1 x N_slow

    % Step 6: Combine all components
    % y[n,m] = A * p_shifted[n] * exp(j*(phase_doppler[m] - phi0))

    % Create phase matrix: subtract phi0 from each Doppler phase
    phase_total = phase_doppler - phi0; % 1 x N_slow

    % Generate complex exponential
    exp_phase = exp(1j * phase_total); % 1 x N_slow

    % Combine: p_shifted (N_fast x 1) with exp_phase (1 x N_slow)
    % Broadcasting: (N_fast x 1) * (1 x N_slow) -> (N_fast x N_slow)
    S_model = A * p_shifted * exp_phase; % N_fast x N_slow
end

function x_noisy = add_complex_noise(x, noise_power)
    % add_complex_noise - Adds complex Gaussian noise CN(0, sigma^2) to signal
    %
    % Inputs:
    %   x           : Input signal (N x M)
    %   noise_power : Noise power sigma^2 (scalar)
    %
    % Output:
    %   x_noisy : Noisy signal x + n (N x M)
    
    [N, M] = size(x);
    
    % Generate complex Gaussian noise CN(0, sigma^2)
    % For CN(0, sigma^2): real and imaginary parts are each N(0, sigma^2/2)
    noise_real = sqrt(noise_power/2) * randn(N, M);
    noise_imag = sqrt(noise_power/2) * randn(N, M);
    
    n = noise_real + 1i * noise_imag;
    
    % Add noise to signal
    x_noisy = x + n;
    
end

% Helper functions for Doppler and wavelength
function dop = speed2dop(speed, wavelength)
    dop = speed / wavelength;
end

function wavelength = freq2wavelen(freq)
    wavelength = physconst('LightSpeed') / freq;
end

function seq = helperMLS(p)
    pol = gfprimdf(p, 2);
    seq = zeros(2^p - 1, 1);
    seq(1:p) = randi([0 1], p, 1);
    for i = (p + 1):(2^p - 1)
        seq(i) = mod(-pol(1:p)*seq(i-p : i-1), 2);
    end
    seq(seq == 0) = -1;
end




