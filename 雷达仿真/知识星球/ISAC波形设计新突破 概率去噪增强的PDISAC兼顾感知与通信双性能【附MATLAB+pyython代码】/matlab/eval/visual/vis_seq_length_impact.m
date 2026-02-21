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

R_data = load('results/Numerical_r_tars_seq_impact.mat');
V_data = load('results/Numerical_v_tars_seq_impact.mat');


Numerical_r_tars = R_data.Numerical_r_tars;
Numerical_v_tars = V_data.Numerical_v_tars;


% Compute MSE
method_names = ["Nprbs255_Nsym256", "Nprbs255_Nsym2560", "Nprbs25500_Nsym256"];

label_names = {
    '$N_{\rm prbs} = 255, N_{\rm sym} = 256$', ...
    '$N_{\rm prbs} = 255, N_{\rm sym} = 2560$', ...
    '$N_{\rm prbs} = 25500, N_{\rm sym} = 256$', ...
};



SNRs_Tx = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30];
CRLB_theoretical = zeros(length(SNRs_Tx), 2);

time_step = time_steps(1);
target_distance = tar_distances(:, 1);
ue_distance = ue_distances(:, 1);
target_speed = tar_speeds(:, 1);
ue_speed = ue_speeds(:, 1);


NF_lin = 10^(NF / 10);   % make separate variable, avoid overwriting NF
Gtx_lin = 10^(Gtx / 10); % Tx gain (linear)
Grx_lin = 10^(Grx / 10); % Rx gain (linear)
noise_power = physconst('Boltzmann') * Tref * NF_lin * B;

NMSE_results = struct();
for m = 1:length(method_names)
    NMSE_results.(method_names(m)).range    = zeros(length(SNRs_Tx), num_of_targets);
    NMSE_results.(method_names(m)).velocity = zeros(length(SNRs_Tx), num_of_targets);
end

for idx_SNR = 1:length(SNRs_Tx)
    for index_method = 1:numel(method_names)
        % Compute Range VAR
        nmse_r = compute_mse( ...
            reshape(Numerical_r_tars(1,idx_SNR,:,:,index_method), ...
                    size(Numerical_r_tars(1, idx_SNR,:,:,index_method),1), []), ...
            num_of_targets, ...
            [target_distance], Rmax);
    
        % Compute Velocity VAR
        nmse_v = compute_mse( ...
            reshape(Numerical_v_tars(1,idx_SNR,:,:,index_method), ...
                    size(Numerical_v_tars(1,idx_SNR,:,:,index_method),1), []), ...
            num_of_targets, ...
            [target_speed], vrelmax);
    
        % Save in struct with method name as key
        NMSE_results.(method_names(index_method)).range(idx_SNR, :) =  nmse_r;
        NMSE_results.(method_names(index_method)).velocity(idx_SNR, :) =  nmse_v;
    end
end


plot_results(SNRs_Tx, NMSE_results, label_names)

function fig = plot_results(SNRsdb, results_struct, label_names)

    % Set default interpreter to LaTeX for all text
    set(groot, 'DefaultTextInterpreter', 'latex');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'DefaultLegendInterpreter', 'latex');
    set(groot, 'DefaultColorbarTickLabelInterpreter', 'latex');

    % Get method names
    method_names = fieldnames(results_struct);
    num_methods = numel(method_names);

    colors = [
        0.0000, 0.4470, 0.7410; % Blue
        0.8500, 0.3250, 0.0980; % Red-orange
        0.9290, 0.6940, 0.1250; % Yellow-orange
        0.4940, 0.1840, 0.5560; % Purple
        0.4660, 0.6740, 0.1880; % Green
        0.3010, 0.7450, 0.9330; % Light blue
        0.6350, 0.0780, 0.1840; % Dark red
        0.0000, 0.5000, 0.0000; % Dark green
        0.7500, 0.0000, 0.7500; % Magenta
        0.0000, 0.0000, 0.0000; % Black
    ];

    line_styles = {'-', '-', '-', '-', '-', '-', '-', '-', '-', '-'};
    markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', '*'};

    if num_methods > size(colors, 1)
        colors = repmat(colors, ceil(num_methods/size(colors,1)), 1);
    end


    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 10, 4]);

    % Set paper size for IEEE format
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [7, 4.2]);

    % Create tiledlayout with IEEE-appropriate spacing
    t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % Marker spacing for clarity
    markerIndices = 1:max(1,floor(length(SNRsdb)/8)):length(SNRsdb);

    % === Left tile: Range plot ===
    ax1 = nexttile(1);
    hold(ax1, 'on');
    h_range = gobjects(num_methods, 1);

    for m = 1:num_methods
        ydata = mean(results_struct.(method_names{m}).range, 2);
        h_range(m) = semilogy(ax1, SNRsdb, ydata, ...
            'LineStyle', line_styles{mod(m-1, length(line_styles))+1}, ...
            'Color', colors(m,:), ...
            'LineWidth', 1.5, ...
            'Marker', markers{mod(m-1, length(markers))+1}, ...
            'MarkerFaceColor', colors(m,:), ...
            'MarkerEdgeColor', colors(m,:), ...
            'MarkerSize', 6, ...
            'MarkerIndices', markerIndices, ...
            'DisplayName', sprintf('%s', label_names{m}));
    end


    ax1.YScale = 'log';
    ax1.FontName = 'Times New Roman';
    ax1.FontSize = 10;
    ax1.LineWidth = 1;
    ax1.Box = 'on';
    ax1.XColor = [0 0 0];
    ax1.YColor = [0 0 0];
    ax1.TickDir = 'in';
    ax1.TickLength = [0.01 0.025];

    grid(ax1, 'on');
    ax1.GridLineStyle = ':';
    ax1.GridAlpha = 0.3;
    ax1.GridColor = [0.15 0.15 0.15];
    ax1.MinorGridLineStyle = 'none';

    ax1.Color = [1 1 1];
    xlim(ax1, [min(SNRsdb) max(SNRsdb)]);
    ylim_range = ylim(ax1);
    ylim(ax1, [ylim_range(1)*0.5, ylim_range(2)*2]);


    % === Right tile: Velocity plot ===
    ax2 = nexttile(2);
    hold(ax2, 'on');
    h_vel = gobjects(num_methods, 1);

    for m = 1:num_methods

        ydata = mean(results_struct.(method_names{m}).velocity, 2);
 
        h_vel(m) = semilogy(ax2, SNRsdb, ydata, ...
            'LineStyle', line_styles{mod(m-1, length(line_styles))+1}, ...
            'Color', colors(m,:), ...
            'LineWidth', 1.5, ...
            'Marker', markers{mod(m-1, length(markers))+1}, ...
            'MarkerFaceColor', colors(m,:), ...
            'MarkerEdgeColor', colors(m,:), ...
            'MarkerSize', 6, ...
            'MarkerIndices', markerIndices, ...
            'DisplayName', sprintf('%s', label_names{m}));
    end

    ax2.YScale = 'log';
    ax2.FontName = 'Times New Roman';
    ax2.FontSize = 10;
    ax2.LineWidth = 1;
    ax2.Box = 'on';
    ax2.XColor = [0 0 0];
    ax2.YColor = [0 0 0];
    ax2.TickDir = 'in';
    ax2.TickLength = [0.01 0.025];


    grid(ax2, 'on');
    ax2.GridLineStyle = ':';
    ax2.GridAlpha = 0.3;
    ax2.GridColor = [0.15 0.15 0.15];
    ax2.MinorGridLineStyle = 'none';


    ax2.Color = [1 1 1];
    xlim(ax2, [min(SNRsdb) max(SNRsdb)]);
    ylim_vel = ylim(ax2);
    ylim(ax2, [ylim_vel(1)*0.5, ylim_vel(2)*2]);


    xlabel(t, 'SNR (dB)', 'FontName', 'Times New Roman', ...
        'FontSize', 11, 'FontWeight', 'normal', 'Interpreter', 'latex');
    ylabel(t, "NMSE", 'FontName', 'Times New Roman', ...
        'FontSize', 11, 'FontWeight', 'normal', 'Interpreter', 'latex');


    % leg = legend(ax2, h_vel, 'Location', 'eastoutside', ...
    %     'Orientation', 'vertical', 'FontName', 'Times New Roman', ...
    %     'FontSize', 9, 'NumColumns', 1);
    % leg.Box = 'on';
    % leg.EdgeColor = [0 0 0];
    % leg.LineWidth = 0.5;
    % leg.Color = [1 1 1];

    leg = legend(ax1, h_range, 'Location', 'layout', ...
        'Orientation', 'horizontal', 'FontName', 'Times New Roman', ...
        'FontSize', 9, 'NumColumns', num_methods);
    leg.Box = 'on';
    leg.EdgeColor = [0 0 0];
    leg.LineWidth = 0.5;
    leg.Color = [1 1 1];
    leg.Layout.Tile = 'north'; % Position at top center
    
    % Add subplot captions below each plot (outside plot area)
    title(ax1, 'NMSE of Range', ...
        'FontSize', 10, ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal', ...
        'Interpreter', 'latex');
    
    title(ax2, 'NMSE of Velocity', ...
        'FontSize', 10, ...
        'FontName', 'Times New Roman', ...
        'FontWeight', 'normal', ...
        'Interpreter', 'latex');


    t.Title.Interpreter = 'none';

    % Set all fonts to Times New Roman (IEEE standard)
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Times New Roman');

    % Configure for high-quality export
    set(fig, 'Renderer', 'painters'); % Vector graphics for publication

    filename = "MSE_Resolution";
    
    % Set figure properties for high-quality PDF export
    set(fig, 'Units', 'Inches');
    pos = get(fig, 'Position');
    set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    
    print(fig, filename, '-dpdf', '-vector', '-r300', '-fillpage');
end