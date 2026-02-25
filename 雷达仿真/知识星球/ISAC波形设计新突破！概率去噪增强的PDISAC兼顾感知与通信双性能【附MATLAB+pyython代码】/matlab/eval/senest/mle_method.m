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
p_prbs = ifft(load("data/p_freq_255.mat").data);
% p_prbs = 2*randi([0,1], num_of_chips, 1) - 1;
P_prbs = fft(p_prbs);

T_d = 5.1e-6;
T_chip = T_d / (2 * num_of_chips);
T_prbs = T_d / 2;
T_D = T_d * num_of_bits;
T_frame = T_D / 2;
fs = 1 / T_chip;
B = fs;



root_dir = 'data/osm/';


num_of_trials = 1000;
SNRs_Tx = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50];



% Preallocate arrays
num_methods = 7;
num_of_targets = 1;
Numerical_r_tars = zeros(length(time_steps), length(SNRs_Tx), num_of_trials, num_of_targets, num_methods);
Numerical_v_tars = zeros(length(time_steps), length(SNRs_Tx), num_of_trials, num_of_targets, num_methods);



Gtx_lin = 10^(Gtx / 10); % Tx gain (linear)
Grx_lin = 10^(Grx / 10); % Rx gain (linear)
NF_lin = 10^(NF / 10);
noise_power = physconst('Boltzmann') * Tref * NF_lin * B;


source_dir = "data/sen/";



for idx_SNR = 1:length(SNRs_Tx)
    SNR_tx = SNRs_Tx(idx_SNR);
    SNR_tx
    Pt = noise_power * 10^(SNR_tx/10);

    for idx_time = 1:length(time_steps)
        time_step = time_steps(idx_time);
        target_distance = tar_distances(:, idx_time);
        ue_distance = ue_distances(:, idx_time);
        target_speed = tar_speeds(:, idx_time);
        ue_speed = ue_speeds(:, idx_time);

        
        alpha_tar =  sqrt((Pt * Gtx_lin * Grx_lin * lambda^2 * target_rcs) / ((4*pi)^3 * target_distance^4));
        alpha_ue =  sqrt((Pt * Gtx_lin * Grx_lin * lambda^2 * target_rcs) / ((4*pi)^3 * ue_distance^4));
        P_sensing = alpha_tar^2;

        root_dir_eval = "evals/freq/";
        % Z_PRBS_waveform_DnCNN_Y_afm = load(fullfile(root_dir_eval, 'DnCNN_Y/afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        % Z_PRBS_waveform_DnCNN_Y_no_afm = load(fullfile(root_dir_eval, 'DnCNN_Y/no_afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        Z_PRBS_waveform_DnCNN_YS_afm = load(fullfile(root_dir_eval, 'DnCNN_YS/afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        Z_PRBS_waveform_DnCNN_YS_no_afm = load(fullfile(root_dir_eval, 'DnCNN_YS/no_afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        Z_PRBS_waveform_PDNet_Y_afm = load(fullfile(root_dir_eval, 'PDNet_Y/afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        % Z_PRBS_waveform_PDNet_Y_no_afm = load(fullfile(root_dir_eval, 'PDNet_Y/no_afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        Z_PRBS_waveform_PDNet_YS_afm = load(fullfile(root_dir_eval, 'PDNet_YS/afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 
        Z_PRBS_waveform_PDNet_YS_no_afm = load(fullfile(root_dir_eval, 'PDNet_YS/no_afm/data/', num2str(time_step), num2str(SNR_tx), '/data.mat')).complex_Z_PRBS_waveforms_pred; 


        for idx_trial = 1:num_of_trials
            % Build file paths
            mat_file_Y_PRBS_waveform = fullfile(root_dir, "Y_PRBS_waveform",  num2str(time_step), num2str(SNR_tx), [num2str(idx_trial) '.mat']);
            
            % Load .mat files
            Y_PRBS_waveform = load(mat_file_Y_PRBS_waveform, 'data').data;
            Z_PRBS_waveform = Y_PRBS_waveform .* conj(P_prbs);           

            % Matched Filter + MLE
            [hat_r_tars, hat_v_tars] = est_rv_mle_y(ifft(Y_PRBS_waveform), p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 1) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 1) = hat_v_tars;

            % MLE
            [hat_r_tars, hat_v_tars] = est_rv_mle_z(ifft(Z_PRBS_waveform), p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 2) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 2) = hat_v_tars;


            % PDNET: YS + AFM + MLE
            z_PRBS_waveform_PDNet_YS_afm = ifft(squeeze(Z_PRBS_waveform_PDNet_YS_afm(idx_trial, :, :)));
            [hat_r_tars, hat_v_tars] = est_rv_mle_z(z_PRBS_waveform_PDNet_YS_afm, p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 3) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 3) = hat_v_tars;


            % PDNET: YS + No AFM + MLE
            z_PRBS_waveform_PDNet_YS_no_afm = ifft(squeeze(Z_PRBS_waveform_PDNet_YS_no_afm(idx_trial, :, :)));
            [hat_r_tars, hat_v_tars] = est_rv_mle_z(z_PRBS_waveform_PDNet_YS_no_afm, p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 4) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 4) = hat_v_tars;


            % PDNET: Y + AFM + MLE
            z_PRBS_waveform_PDNet_Y_afm = ifft(squeeze(Z_PRBS_waveform_PDNet_Y_afm(idx_trial, :, :)));
            [hat_r_tars, hat_v_tars] = est_rv_mle_z(z_PRBS_waveform_PDNet_Y_afm, p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 5) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 5) = hat_v_tars;

            % DnCNN: YS + AFM + MLE
            z_PRBS_waveform_DnCNN_YS_afm = ifft(squeeze(Z_PRBS_waveform_DnCNN_YS_afm(idx_trial, :, :)));
            [hat_r_tars, hat_v_tars] = est_rv_mle_z(z_PRBS_waveform_DnCNN_YS_afm, p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 6) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 6) = hat_v_tars;


            % DnCNN: YS + No AFM + MLE
            z_PRBS_waveform_DnCNN_YS_no_afm = ifft(squeeze(Z_PRBS_waveform_DnCNN_YS_no_afm(idx_trial, :, :)));
            [hat_r_tars, hat_v_tars] = est_rv_mle_z(z_PRBS_waveform_DnCNN_YS_no_afm, p_prbs, noise_power, P_sensing, target_distance, target_speed, fc, T_chip, T_d);
            Numerical_r_tars(idx_time, idx_SNR, idx_trial, :, 7) = hat_r_tars;
            Numerical_v_tars(idx_time, idx_SNR, idx_trial, :, 7) = hat_v_tars;


            % rdr = phased.RangeDopplerResponse(...
            %     'RangeMethod', 'FFT', ...
            %     'SampleRate', fs, ...
            %     'SweepSlope', -B/T_prbs, ...
            %     'DopplerOutput', 'Speed', ...
            %     'OperatingFrequency', fc, ...
            %     'PRFSource', 'Property', ...
            %     'PRF', 1/T_d, ...
            % 'ReferenceRangeCentered', false);
            % figure;
            % plotResponse(rdr, fft(z_PRBS_waveform_DnCNN_YS_no_afm), 'Unit', 'db');
            % xlim([-vrelmax vrelmax]);
            % ylim([0 Rmax]);
            % hold on;
            % scatter(hat_v_tars, hat_r_tars, 100, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
            % scatter([target_speed], [target_distance], 100, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
            % hold off;
            
        end
    end
end

% Save the results
save('results/Numerical_r_tars_mle_method.mat', 'Numerical_r_tars', '-v7.3');
save('results/Numerical_v_tars_mle_method.mat', 'Numerical_v_tars', '-v7.3');
