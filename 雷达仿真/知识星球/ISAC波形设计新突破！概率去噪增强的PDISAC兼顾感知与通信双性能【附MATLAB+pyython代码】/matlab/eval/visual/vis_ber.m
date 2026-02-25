rng("default");

siteviewer( ...
    Buildings="/data/polytechnique.osm", ...
    Basemap="satellite");


fc = 24e9; % Carrier frequency (Hz)

radar_position = [45.501361,-73.614219, 155]; % lat, lon, alt
radar_array_orientation = [0 0].';       % azimuth (0 deg is East, 90 deg is North) and elevation (positive points upwards) in deg
radar_vel = [0; 0; 0];
radar_location = [0; 0; 0];
radar_motion = phased.Platform('InitialPosition', radar_location, 'Velocity', radar_vel);

ue_position = [45.501421,-73.614125, 155];
% ue_position = [45.501511,-73.613966, 155];
% ue_position = [45.501543,-73.613588, 150];
% ue_position = [45.501361,-73.614219, 154];
% ue_position = [45.501392,-73.614180, 154];
ue_array_orientation = [0 0].';       % azimuth (0 deg is East, 90 deg is North) and elevation (positive points upwards) in deg
ue_vel = [-3; -6; 0];
% ue_vel = [-30; -40; 0];
ue_reflections_order = 1;                 % number of reflections for ray tracing analysis (0 for LOS)
[x, y, z] = latlon2local(ue_position(1), ue_position(2), ue_position(3), radar_position);
ue_location = [x; y; z];  % Local position of the target
ue_motion = phased.Platform('InitialPosition', ue_location, 'Velocity', ue_vel);
ue_distance = vecnorm(ue_location);
ue_rcs = 1.3;


radarSite = txsite("Name","Radar", ...
    "Latitude",radar_position(1),"Longitude",radar_position(2),...
    "AntennaAngle",radar_array_orientation(1:2),...
    "AntennaHeight",3,...  % in m
    "TransmitterFrequency",fc);

ueSite = rxsite("Name","UE", ...
    "Latitude",ue_position(1),"Longitude",ue_position(2),...
    "AntennaHeight",1,... % in m
    "AntennaAngle",ue_array_orientation(1:2));

% Compute rays at ue
pm = propagationModel("raytracing","Method","sbr","MaxNumReflections",ue_reflections_order,"MaxNumDiffractions",0);
uerays = raytrace(radarSite,ueSite,pm,"Type","pathloss");

show(radarSite);
show(ueSite);
plot(uerays{1});


% Signal Settings
c_prbs = 7;
p_prbs = helperMLS(c_prbs);
num_of_bits = 1;
num_of_chips = 2^c_prbs - 1;

% Parameters
fc = 24e9; % Carrier frequency (Hz)
B = 100e6; % Bandwidth (Hz)
fs = B; % Sample rate equal to bandwidth
Gtx = 50; % Tx antenna gain (dB)
Gtx_lin = 10^(Gtx / 10); % Tx gain (linear)

Grx_ue = 30; % UE Rx antenna gain (dB)
Grx_ue_lin = 10^(Grx_ue / 10); % Rx gain (linear)

NF = 2.9; % Noise figure (dB)
NF_lin = 10^(NF / 10);

Tref = 290; % Reference temperature (K)
Rmax = 200; % Maximum range of interest
vrelmax = 60; % Maximum relative velocity
c = physconst('LightSpeed');
lambda = c / fc;
eps_val = 1e-12;

% Duration calculations
T_chip = 1/B;            % Chip duration
T_bits = num_of_chips * T_chip;    % Modulation period
T_bpsk = T_bits;
T_prbs = T_bpsk;
T_d_isac = 2*T_prbs;
T_D_isac = T_d_isac * num_of_bits;
T_frame = T_D_isac / 2;


num_of_trials = 15000;

% Communication results storage (now two numerical BERs: estimated and perfect)
SNRs_Tx = [-20, -17, -15, -12 -10, -7, -5, -2, 0, 2,  5, 7, 10, 12, 15];
Theoretical_BERs = zeros(length(SNRs_Tx), 1);
Theoretical_bergading_BERs = zeros(length(SNRs_Tx), 1);
Numerical_BERs_est = zeros(length(SNRs_Tx), 1);   % existing estimator result
Numerical_BERs_perf = zeros(length(SNRs_Tx), 1);  % perfect CSI got from LS method without noises
Numerical_BERs_perf_v2 = zeros(length(SNRs_Tx), 1);  % perfect CSI got from real channel
Numerical_Capacity_est = zeros(length(SNRs_Tx), 1);   % existing estimator result
Numerical_Capacity_csi = zeros(length(SNRs_Tx), 1);  % perfect CSI got from LS method without noises
Numerical_Capacity_csi_v2 = zeros(length(SNRs_Tx), 1);  % perfect CSI got from real channel

ant_ue = phased.IsotropicAntennaElement;
radiator_ue = phased.Radiator('Sensor', ant_ue, 'OperatingFrequency', fc);
collector_ue = phased.Collector('Sensor', ant_ue, 'OperatingFrequency', fc);
receiver_ue = phased.ReceiverPreamp('SampleRate', fs, 'Gain', Grx_ue, 'NoiseFigure', NF, 'ReferenceTemperature', Tref);
radar_to_ue_channel = phased.FreeSpace('SampleRate', fs, 'TwoWayPropagation', false, 'OperatingFrequency', fc);
receiver_ue_no_noise = phased.ReceiverPreamp( ...
    'Gain', Grx_ue, ...
    'NoiseMethod', 'Noise power', ... 
    'NoisePower', 0); 

% Rician channel
path_delays = [uerays{1}.PropagationDelay]-min([uerays{1}.PropagationDelay]);
average_path_gains = -[uerays{1}.PathLoss];
average_path_gains_lin = 10.^(average_path_gains/10);
G_multipath = sum(average_path_gains_lin);
K = average_path_gains_lin(1) / sum(average_path_gains_lin(2:end)); % Linear K
fdmax = 2 * speed2dop(vecnorm(ue_vel), freq2wavelen(fc));
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
    'Seed', 12345); 


% compareRicianDistribution(clone(commChannel), [], 10e5)


noise_power = physconst('Boltzmann') * Tref * NF_lin * B;
for idx_SNR = 1:length(SNRs_Tx)
    SNR_tx = SNRs_Tx(idx_SNR);


    Pt = noise_power * 10^(SNR_tx/10);
    P_scaled = (Gtx_lin * Grx_ue_lin * lambda^2) / ((4*pi)^2 * ue_distance^2);
    bar_SNR_Rx_dB = SNR_tx + 10*log10(P_scaled);



    % Radar components setup for ue (Radar -> UE -> Radar)
    transmitter_ue = phased.Transmitter('Gain', Gtx, 'PeakPower', Pt);
    
    
    total_errors_est = 0;     % errors using estimated channel
    total_errors_perf = 0;    % errors using perfect CSI (without noises)
    total_errors_perf_v2 = 0; % errors using real channel

    capacity_avg_est = 0;
    capacity_avg_csi = 0;
    capacity_avg_csi_v2 = 0;

    received_symbol_power_est = 0;
    received_symbol_power_perf = 0;
    received_symbol_power_perf_v2 = 0;
    noise_power_est = 0;
    noise_power_perf = 0;
    noise_power_perf_v2 = 0;


    % Main monte-carlo loop
    for idx_trial = 1:num_of_trials
        bits = randi([0, 1], [num_of_bits, 1]);
        s_pbsk = pskmod(bits, 2);
        
        M_prbs = p_prbs * ones(1, num_of_bits);
        M_pmcw = p_prbs * s_pbsk.';
        D_isac = [M_prbs; M_pmcw];

        % Initialize output
        y_C_isac = zeros(size(D_isac));      % after receiver (contains noise)
        rx_prop_all = zeros(size(D_isac));   % store channel output BEFORE receiver (noise-free)
        H_perfect_all = zeros(num_of_chips, num_of_bits); % Store perfect CSI for each symbol

        reset(radar_motion);
        reset(ue_motion);
        
        impulse_signal = zeros(num_of_chips, 1);
        impulse_signal(1) = 1;
    
        % Radar processing: transmit each ISAC symbol, propagate, collect
        for m = 1:num_of_bits
            % Advance platform and get positions (single call with positive time)
            [cur_l_radar, cur_v_radar] = radar_motion(T_d_isac);
            [cur_l_ue,    cur_v_ue]    = ue_motion(T_d_isac);
            [cur_r_ue, cur_a_ue] = rangeangle(cur_l_ue, cur_l_radar);



            ue_sig_input = D_isac(:, m); 
            ue_sig = transmitter_ue(ue_sig_input);
            ue_radsig = radiator_ue(ue_sig, cur_a_ue);
            ue_forward_output = radar_to_ue_channel(ue_radsig, cur_l_radar, cur_l_ue, cur_v_radar, cur_v_ue);

            
            [ue_forward_output_rician, h_rician] = commChannel(ue_forward_output);
            

            rx_prop_all(:, m) = receiver_ue_no_noise(ue_forward_output_rician);
            y_C_isac(:, m) = receiver_ue(ue_forward_output_rician);

            X_perfect = fft(ue_sig_input(num_of_chips+1:end,1));
            Y_perfect = fft(rx_prop_all(num_of_chips+1:end,m));
            H_perfect_all(:, m) = Y_perfect ./ X_perfect;

  
        end
        % ##############################
        % # Communication Post Process #
        % ##############################

        P_prbs = fft(p_prbs); % P(f)


        % Channel estimation
        y_C_isac = reshape(y_C_isac, 2*num_of_chips, []);      % (2Nchips x Nsymbols)
        y_PRBS_waveform =  y_C_isac(1:num_of_chips, :);        % received PRBS (with noise)
        Y_PRBS_waveform = fft(y_PRBS_waveform);                % FFT along columns (default)

        H_PRBS_waveform = Y_PRBS_waveform ./ P_prbs; % estimated H (Nfft x Nsymbols)
        
        % Matched filter & equalization using estimated CSI (your existing pipeline)
        y_ISAC_waveform = y_C_isac(num_of_chips+1:end, :);
        Y_ISAC_waveform = fft(y_ISAC_waveform);

        Z_ISAC_waveform = Y_ISAC_waveform .* conj(P_prbs);
        Z_ISAC_waveform = Z_ISAC_waveform ./ (H_PRBS_waveform);
        z_ISAC_waveform = ifft(Z_ISAC_waveform);

        [~, idxmax] = max(abs(z_ISAC_waveform), [], 'linear');
        z_bpsk_waveform = z_ISAC_waveform(idxmax).';
        hat_s_pbsk_est = z_bpsk_waveform./num_of_chips;
        hat_bits_est = pskdemod(hat_s_pbsk_est, 2);
        total_errors_est = total_errors_est + sum(hat_bits_est ~= bits);   % accumulate errors

     

        % Channel estimation with LS given the observation without noises
        % Use the PRBS portion of the noise-free channel output (rx_prop_all)
        rx_prop_all = reshape(rx_prop_all, 2*num_of_chips, []);
        y_PRBS_prop = rx_prop_all(1:num_of_chips, :);   % noise-free PRBS waveform (before receiver)
        Y_PRBS_prop = fft(y_PRBS_prop);
        H_PRBS_true = Y_PRBS_prop ./ P_prbs;   % exact H used on transmitted signal

        % Matched filter & equalization
        % Re-use Y_ISAC_waveform (FFT of receiver output PMCW) because PMCW was fed through same channel
        Z_ISAC_waveform_true = Y_ISAC_waveform .* conj(P_prbs);
        Z_ISAC_waveform_true = Z_ISAC_waveform_true ./ (H_PRBS_true);
        z_ISAC_waveform_true = ifft(Z_ISAC_waveform_true);

        [~, idxmax_t] = max(abs(z_ISAC_waveform_true), [], 'linear');
        z_bpsk_waveform_true = z_ISAC_waveform_true(idxmax_t).';
        hat_s_pbsk_perf = z_bpsk_waveform_true./num_of_chips;
        hat_bits_perf = pskdemod(hat_s_pbsk_perf, 2);
        total_errors_perf = total_errors_perf + sum(hat_bits_perf ~= bits);

        

        % Matched filter & equalization with TRUE perfect CSI
        Z_ISAC_waveform_true_v2 = Y_ISAC_waveform .* conj(P_prbs);
        Z_ISAC_waveform_true_v2 = Z_ISAC_waveform_true_v2 ./ H_perfect_all;  % Use H_perfect_all!
        
        z_ISAC_waveform_true_v2 = ifft(Z_ISAC_waveform_true_v2);
        
        [~, idxmax_t_v2] = max(abs(z_ISAC_waveform_true_v2), [], 'linear');
        z_bpsk_waveform_true_v2 = z_ISAC_waveform_true_v2(idxmax_t_v2).';
        hat_s_pbsk_perf_v2 = z_bpsk_waveform_true_v2 ./ num_of_chips;
        hat_bits_perf_v2 = pskdemod(hat_s_pbsk_perf_v2, 2);
        total_errors_perf_v2 = total_errors_perf_v2 + sum(hat_bits_perf_v2 ~= bits);
        


        % ====================================================================
        % Capacity Calculation
        % ====================================================================
        

        % Capacity based on perfect CSI
        SNR_per_bin_csi_v2 = (abs(H_perfect_all).^2 .* abs(P_prbs).^2) / (noise_power * num_of_chips);
        capacity_instant_csi_v2 = sum(log2(1 + SNR_per_bin_csi_v2), 'all') / num_of_chips;
        capacity_avg_csi_v2 = capacity_avg_csi_v2 + capacity_instant_csi_v2;

        % Capacity based on LS channel without noises
        error_var = abs((H_PRBS_true  - H_perfect_all).* P_prbs).^2;
        SNR_per_bin_csi = (abs(H_perfect_all).^2 .* abs(P_prbs).^2) ./ (noise_power * num_of_chips + error_var);
        capacity_instant_csi = sum(log2(1 + SNR_per_bin_csi), 'all') / num_of_chips;
        capacity_avg_csi = capacity_avg_csi + capacity_instant_csi;


        % Capacity based on LS channel with noises
        error_var = abs((H_PRBS_waveform  - H_perfect_all).* P_prbs).^2;
        SNR_per_bin_est = (abs(H_perfect_all).^2 .* abs(P_prbs).^2) ./ (noise_power * num_of_chips + error_var);
        capacity_instant_est = sum(log2(1 + SNR_per_bin_est), 'all') / num_of_chips;
        capacity_avg_est = capacity_avg_est + capacity_instant_est;


    end


    % ####################
    % # BER in numerical #
    % ####################
    Numerical_BERs_est(idx_SNR, 1)  =  total_errors_est / (num_of_trials * num_of_bits);
    Numerical_BERs_perf(idx_SNR, 1) =  total_errors_perf / (num_of_trials * num_of_bits);
    Numerical_BERs_perf_v2(idx_SNR, 1) =  total_errors_perf_v2 / (num_of_trials * num_of_bits);

    % ######################
    % # BER in theoretical #
    % ######################
    
    % P_scaled = (Gtx_lin * Grx_ue_lin * lambda^2) / ((4*pi)^2 * ue_distance^2);
    bar_SNR_Rx_linear = 10.^(bar_SNR_Rx_dB/10);
    
    integrand = @(theta) ((1 + K) ./ (1 + K + bar_SNR_Rx_linear./(sin(theta).^2))) .* ...
            exp(-K * bar_SNR_Rx_linear ./ ((1 + K)*sin(theta).^2 + bar_SNR_Rx_linear));
    Theoretical_BERs(idx_SNR, 1) = (1/pi) * integral(integrand, 0, pi/2);
    Theoretical_bergading_BERs(idx_SNR, 1) = berfading(10 * log10(bar_SNR_Rx_linear), 'psk', 2, 1, K);

    % #########################
    % # Capacity in numerical #
    % #########################
    Numerical_Capacity_est(idx_SNR, 1) =  capacity_avg_est / (num_of_trials * num_of_bits);
    Numerical_Capacity_csi(idx_SNR, 1) =  capacity_avg_csi / (num_of_trials * num_of_bits);
    Numerical_Capacity_csi_v2(idx_SNR, 1) =  capacity_avg_csi_v2 / (num_of_trials * num_of_bits);

    
    fprintf('SNR: %d dB | BER_est: %.6f | BER_perf: %.6f | BER_perf_v2: %.6f | BER_theo: %.6f | Cap_est: %.4f | Cap_perf_v2: %.4f\n', ...
        SNR_tx, Numerical_BERs_est(idx_SNR), Numerical_BERs_perf(idx_SNR), Numerical_BERs_perf_v2(idx_SNR), Theoretical_bergading_BERs(idx_SNR), ...
        Numerical_Capacity_est(idx_SNR), Numerical_Capacity_csi_v2(idx_SNR));
end

% === Plot the three curves: estimated, perfect, theoretical (with processing gain) ===
new_SNRs_Tx = [-20, -17, -15, -12 -10, -7, -5, -2, 0, 2,  5, 7, 10, 12, 15];
figure;
semilogy(SNRs_Tx, Numerical_BERs_est, 'o-','LineWidth',1.6); hold on;
semilogy(SNRs_Tx, Numerical_BERs_perf,'s-','LineWidth',1.6);
semilogy(SNRs_Tx, Numerical_BERs_perf_v2,'d-','LineWidth',1.6);
semilogy(SNRs_Tx, Theoretical_BERs, '^-','LineWidth',1.2);
semilogy(SNRs_Tx, Theoretical_bergading_BERs,'v--','LineWidth',1.2);
grid on; box on;
xlabel('SNR per sample (dB)');
ylabel('BER');
legend('Numerical BER (LS with noise)', ...
       'Numerical BER (LS without noise)', ...
       'Numerical BER (perfect CSI)', ...
       'Theoretical BER (MGF func)', ...
       'Theoretical BER (erfading)', ...
       'Location','southwest');
title('BER Comparison - Corrected Perfect CSI');

figure;
plot(SNRs_Tx, Numerical_Capacity_est, 'o-','LineWidth',1.6); hold on;
plot(SNRs_Tx, Numerical_Capacity_csi,'s-','LineWidth',1.6);
plot(SNRs_Tx, Numerical_Capacity_csi_v2,'d-','LineWidth',1.6);
grid on; box on;
xlabel('SNR per sample (dB)');
ylabel('Capacity (bit/s/Hz)');
legend('Numerical Capacity (LS with noise)', ...
       'Numerical Capacity (LS without noise)', ...
       'Numerical Capacity (perfect CSI)', ...
       'Location','northwest');
title('Capacity Comparison - Corrected Perfect CSI');


% % Create a structure with your variables
% % Define the filename
% filename = 'results/simulation_results.mat';
% 
% % Create the new entry once
% new_entry = struct();
% new_entry.ue_distance = ue_distance;
% new_entry.SNRs_Tx = SNRs_Tx;
% new_entry.Numerical_BERs_est = Numerical_BERs_est;
% new_entry.Numerical_BERs_perf = Numerical_BERs_perf;
% new_entry.Numerical_BERs_perf_v2 = Numerical_BERs_perf_v2;
% new_entry.Theoretical_BERs = Theoretical_BERs;
% new_entry.Theoretical_bergading_BERs = Theoretical_bergading_BERs;
% new_entry.Numerical_Capacity_est = Numerical_Capacity_est;
% new_entry.Numerical_Capacity_csi = Numerical_Capacity_csi;
% new_entry.Numerical_Capacity_csi_v2 = Numerical_Capacity_csi_v2;
% 
% if isfile(filename)
%     existing_data = load(filename);
%     field_names = fieldnames(existing_data);
% 
%     overwrite_idx = '';
%     max_idx = 0;
% 
%     % Scan existing entries
%     for i = 1:length(field_names)
%         fname = field_names{i};
%         idx_num = sscanf(fname, 'idx%d');
%         max_idx = max(max_idx, idx_num);
% 
%         if isfield(existing_data.(fname), 'ue_distance')
%             if abs(existing_data.(fname).ue_distance - ue_distance) < 1e-6
%                 overwrite_idx = fname;
%                 break;
%             end
%         end
%     end
% 
%     if ~isempty(overwrite_idx)
%         % Override existing entry
%         existing_data.(overwrite_idx) = new_entry;
%         fprintf('Overwritten ue_distance = %.2f at %s.\n', ue_distance, overwrite_idx);
%     else
%         % Append new entry
%         next_idx = max_idx + 1;
%         existing_data.(sprintf('idx%d', next_idx)) = new_entry;
%         fprintf('Added ue_distance = %.2f at idx%d.\n', ue_distance, next_idx);
%     end
% 
%     save(filename, '-struct', 'existing_data');
% 
% else
%     % File does not exist → create it
%     data_to_save = struct();
%     data_to_save.idx1 = new_entry;
% 
%     save(filename, '-struct', 'data_to_save');
%     fprintf('Created new file with ue_distance = %.2f at idx1.\n', ue_distance);
% end



% new_SNRs_Tx = [-20, -17, -15, -12, -10, -7, -5, 0, 5, 10, 15];
% 
% [~, idx] = ismember(new_SNRs_Tx, SNRs_Tx);
% 
% % Safety check
% idx = idx(idx > 0);
% 
% figure;
% semilogy(new_SNRs_Tx, Numerical_BERs_est(idx), 'o-','LineWidth',1.6); hold on;
% semilogy(new_SNRs_Tx, Numerical_BERs_perf(idx),'s-','LineWidth',1.6);
% semilogy(new_SNRs_Tx, Numerical_BERs_perf_v2(idx),'d-','LineWidth',1.6);
% semilogy(new_SNRs_Tx, Theoretical_BERs(idx), '^-','LineWidth',1.2);
% 
% grid on; box on;
% xlabel('SNR per sample (dB)');
% ylabel('BER');
% legend('LS-Est with imperfect CSI (sim.)', ...
%        'LS-Est with perfect  CSI (sim.)', ...
%        'Perfect Est (sim.)', ...
%        'Our Analysis', ...
%        'Location','southwest');
% 
% % title('BER Comparison - Corrected Perfect CSI');
% xlim([min(new_SNRs_Tx), max(new_SNRs_Tx)]);


% BER Comparison Plot with 2D Line Style

% Set default interpreter to LaTeX for all text
set(groot, 'DefaultTextInterpreter', 'latex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
set(groot, 'DefaultLegendInterpreter', 'latex');

% Data preparation
new_SNRs_Tx = [-20, -17, -15, -12, -10, -7, -5, 0, 5, 10, 15];
[~, idx] = ismember(new_SNRs_Tx, SNRs_Tx);
% Safety check
idx = idx(idx > 0);

% Create figure with IEEE formatting
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 8, 6]);

% Set paper size for IEEE format
set(fig, 'PaperPositionMode', 'auto');
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperSize', [8, 6]);

% Create axes
ax = axes('Parent', fig);
hold(ax, 'on');

% Define color map for different lines
colors = lines(4);

% Arrays to store legend handles
legend_handles = [];
legend_labels = {};

% Plot 1: LS-Est with imperfect CSI
h1 = plot(ax, new_SNRs_Tx, Numerical_BERs_est(idx), ...
    'Color', colors(1,:), ...
    'LineStyle', '-', ...
    'LineWidth', 1.5, ...
    'Marker', 'o', ...
    'MarkerSize', 6, ...
    'MarkerFaceColor', colors(1,:), ...
    'DisplayName', 'LS-Est with imperfect CSI (sim.)');
legend_handles = [legend_handles, h1];
legend_labels{end+1} = 'LS-Est with imperfect CSI (sim.)';

% Plot 2: LS-Est with perfect CSI
h2 = plot(ax, new_SNRs_Tx, Numerical_BERs_perf(idx), ...
    'Color', colors(2,:), ...
    'LineStyle', '-', ...
    'LineWidth', 1.5, ...
    'Marker', 's', ...
    'MarkerSize', 6, ...
    'MarkerFaceColor', colors(2,:), ...
    'DisplayName', 'LS-Est with perfect CSI (sim.)');
legend_handles = [legend_handles, h2];
legend_labels{end+1} = 'LS-Est with perfect CSI (sim.)';

% Plot 3: Perfect Est
h3 = plot(ax, new_SNRs_Tx, Numerical_BERs_perf_v2(idx), ...
    'Color', colors(3,:), ...
    'LineStyle', '-', ...
    'LineWidth', 1.5, ...
    'Marker', 'd', ...
    'MarkerSize', 6, ...
    'MarkerFaceColor', colors(3,:), ...
    'DisplayName', 'Perfect Est (sim.)');
legend_handles = [legend_handles, h3];
legend_labels{end+1} = 'Perfect Est (sim.)';

% Plot 4: Our Analysis
h4 = plot(ax, new_SNRs_Tx, Theoretical_BERs(idx), ...
    'Color', colors(4,:), ...
    'LineStyle', '-', ...
    'LineWidth', 1.5, ...
    'Marker', '^', ...
    'MarkerSize', 6, ...
    'MarkerFaceColor', colors(4,:), ...
    'DisplayName', 'Our Analysis');
legend_handles = [legend_handles, h4];
legend_labels{end+1} = 'Our Analysis';

% Set logarithmic scale for BER (y-axis)
set(ax, 'YScale', 'log');

% Axis labels
xlabel(ax, 'SNR per sample (dB)', 'FontName', 'Times New Roman', ...
    'FontSize', 12, 'Interpreter', 'latex');
ylabel(ax, 'BER', 'FontName', 'Times New Roman', ...
    'FontSize', 12, 'Interpreter', 'latex');

% Set x-axis limits
xlim(ax, [min(new_SNRs_Tx), max(new_SNRs_Tx)]);

% Grid settings
grid(ax, 'on');
ax.GridLineStyle = ':';
ax.GridAlpha = 0.3;
ax.GridColor = [0.15 0.15 0.15];
ax.MinorGridLineStyle = ':';
ax.MinorGridAlpha = 0.15;

% Font and line settings
ax.FontName = 'Times New Roman';
ax.FontSize = 11;
ax.LineWidth = 1;
ax.Box = 'on';
ax.XColor = [0 0 0];
ax.YColor = [0 0 0];

% Legend
leg = legend(ax, legend_handles, legend_labels, ...
    'Location', 'southwest', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 9, ...
    'Interpreter', 'latex');
leg.Box = 'on';
leg.EdgeColor = [0 0 0];
leg.LineWidth = 0.5;
leg.Color = [1 1 1];

% Set all fonts to Times New Roman
set(findall(fig, '-property', 'FontName'), 'FontName', 'Times New Roman');

% Configure for high-quality export
set(fig, 'Renderer', 'painters');

% Export to PDF
filename = 'BER_Comparison';

% Set figure properties for high-quality PDF export
set(fig, 'Units', 'Inches');
pos = get(fig, 'Position');
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
    'PaperSize', [pos(3), pos(4)]);

print(fig, filename, '-dpdf', '-vector', '-r300', '-fillpage');

fprintf('Figure saved as %s.pdf\n', filename);

hold(ax, 'off');



%% Helper functions
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



function compareRicianDistribution(commChannel, numerical_rician_model, Nsamp)
    %COMPARERICIANDISTRIBUTION
    %   Compare MATLAB empirical Rician distribution,
    %   Theoretical Rician PDF,
    %   And user numerical model (if provided).
    %
    % Inputs:
    %   commChannel            : configured comm.RicianChannel object
    %   numerical_rician_model : user's numerical amplitudes OR []
    %   Nsamp                  : number of Monte-Carlo samples from MATLAB channel
    %
    % Example:
    %   compareRicianDistribution(commChannel, numerical_rician_model, 2e6)
    
    % Set default interpreter to LaTeX for all text
    set(groot, 'DefaultTextInterpreter', 'latex');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'DefaultLegendInterpreter', 'latex');
    
    % ===============================================================
    % Handle input
    % ===============================================================
    if nargin < 3
        Nsamp = 2e6;
    end
    if isempty(numerical_rician_model)
        use_user_model = false;
    else
        use_user_model = true;
        % Ensure column vector shape
        numerical_rician_model = numerical_rician_model(:);
    end
    
    % ===============================================================
    % 1. Generate MATLAB empirical channel samples
    % ===============================================================
    x = ones(Nsamp,1);
    [~, pathGains] = commChannel(x);
    
    % LOS path
    h0 = pathGains(:,1);
    r_emp = abs(h0);   % empirical amplitude
    
    % ===============================================================
    % 2. Extract channel parameters (K, powers)
    % ===============================================================
    K = commChannel.KFactor;
    avgGains_dB = commChannel.AveragePathGains(:);
    P = 10.^(avgGains_dB/10);
    P = P / sum(P);     % normalized
    P0 = P(1);
    s  = sqrt( P0 * K/(K+1) );      % LOS amplitude
    sigma = sqrt( P0/(2*(K+1)) );   % scatter std dev
    
    % ===============================================================
    % 3. Compute theoretical Rician PDF
    % ===============================================================
    % Define axis
    r_max = max([r_emp; use_user_model*max(numerical_rician_model)]);
    r_axis = linspace(0, 1.2*r_max, 2000);
    
    % Theoretical Rician PDF
    pdf_th = (r_axis ./ sigma.^2) .* ...
             exp(-(r_axis.^2 + s.^2)/(2*sigma.^2)) .* ...
             besseli(0, (r_axis*s)/(sigma.^2));
    
    % ===============================================================
    % 4. Plot with 2D line style
    % ===============================================================
    % Create figure with IEEE formatting
    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 8, 6]);
    
    % Set paper size for IEEE format
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [8, 6]);
    
    % Create axes
    ax = axes('Parent', fig);
    hold(ax, 'on');
    
    % Define color map (matching previous style)
    colors = lines(3);
    
    % Arrays to store legend handles
    legend_handles = [];
    legend_labels = {};
    
    % Plot empirical as stairs histogram
    h1 = histogram(ax, r_emp, 'Normalization', 'pdf', ...
        'DisplayStyle', 'stairs', ...
        'LineWidth', 1.5, ...
        'EdgeColor', colors(1,:));
    
    % Get histogram data for legend (create dummy line)
    dummy1 = plot(ax, NaN, NaN, '-', 'LineWidth', 1.5, 'Color', colors(1,:), ...
        'DisplayName', 'Simulation Rician');
    legend_handles = [legend_handles, dummy1];
    legend_labels{end+1} = 'Simulation Rician';
    
    % Plot theoretical
    h2 = plot(ax, r_axis, pdf_th, ...
        'Color', colors(2,:), ...
        'LineStyle', '-', ...
        'LineWidth', 1.5, ...
        'DisplayName', 'Our Analysis');
    legend_handles = [legend_handles, h2];
    legend_labels{end+1} = 'Our Analysis';
    
    % Plot user model if exists
    if use_user_model
        h3 = histogram(ax, numerical_rician_model, 'Normalization', 'pdf', ...
            'DisplayStyle', 'stairs', ...
            'LineWidth', 1.5, ...
            'EdgeColor', colors(3,:));
        
        % Create dummy line for legend
        dummy3 = plot(ax, NaN, NaN, '-', 'LineWidth', 1.5, 'Color', colors(3,:), ...
            'DisplayName', 'User Numerical Model');
        legend_handles = [legend_handles, dummy3];
        legend_labels{end+1} = 'User Numerical Model';
    end
    
    % Axis labels
    xlabel(ax, 'Amplitude', 'FontName', 'Times New Roman', ...
        'FontSize', 12, 'Interpreter', 'latex');
    ylabel(ax, 'PDF', 'FontName', 'Times New Roman', ...
        'FontSize', 12, 'Interpreter', 'latex');
    
    % Grid settings
    grid(ax, 'on');
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.3;
    ax.GridColor = [0.15 0.15 0.15];
    ax.MinorGridLineStyle = ':';
    ax.MinorGridAlpha = 0.15;
    
    % Font and line settings
    ax.FontName = 'Times New Roman';
    ax.FontSize = 11;
    ax.LineWidth = 1;
    ax.Box = 'on';
    ax.XColor = [0 0 0];
    ax.YColor = [0 0 0];
    
    % Set axes limits
    xlim(ax, [0, 1.2*r_max]);
    ylim_curr = ylim(ax);
    ylim(ax, [0, ylim_curr(2)*1.05]);
    
    % Legend
    leg = legend(ax, legend_handles, legend_labels, ...
        'Location', 'best', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 9, ...
        'Interpreter', 'latex');
    leg.Box = 'on';
    leg.EdgeColor = [0 0 0];
    leg.LineWidth = 0.5;
    leg.Color = [1 1 1];
    
    % Set all fonts to Times New Roman
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Times New Roman');
    
    % Configure for high-quality export
    set(fig, 'Renderer', 'painters');
    
    % Export to PDF
    filename = 'Rician_Distribution_Comparison';
    
    % Set figure properties for high-quality PDF export
    set(fig, 'Units', 'Inches');
    pos = get(fig, 'Position');
    set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
        'PaperSize', [pos(3), pos(4)]);
    
    print(fig, filename, '-dpdf', '-vector', '-r300', '-fillpage');
    
    fprintf('Figure saved as %s.pdf\n', filename);
    
    hold(ax, 'off');

end