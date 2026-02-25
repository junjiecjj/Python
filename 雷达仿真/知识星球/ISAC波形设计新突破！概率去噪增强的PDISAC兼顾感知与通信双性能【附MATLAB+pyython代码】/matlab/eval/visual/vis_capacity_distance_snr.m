function fig = plot_capacity_3d_surface(filepath)
    % Load the .mat file
    data = load(filepath);
    
    % Set default interpreter to LaTeX for all text
    set(groot, 'DefaultTextInterpreter', 'latex');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'DefaultLegendInterpreter', 'latex');
    set(groot, 'DefaultColorbarTickLabelInterpreter', 'latex');
    
    % Get all field names (excluding internal MATLAB fields)
    keys = fieldnames(data);
    keys = keys(~startsWith(keys, '__'));
    
    % Initialize arrays
    distances = [];
    snrs_all = [];
    caps_csi_perfect_all = [];
    caps_csi_est_all = [];
    
    % Extract data from each structure
    for i = 1:length(keys)
        entry = data.(keys{i});
        
        d = double(entry.ue_distance);
        snrs = double(entry.SNRs_Tx(:));
        
        caps_csi_perfect = double(entry.Numerical_Capacity_csi_v2(:));
        caps_csi_est = double(entry.Numerical_Capacity_est(:));
        
        distances = [distances; repmat(d, length(snrs), 1)];
        snrs_all = [snrs_all; snrs];
        caps_csi_perfect_all = [caps_csi_perfect_all; caps_csi_perfect];
        caps_csi_est_all = [caps_csi_est_all; caps_csi_est];
    end
    
    % Rename for clarity (X = SNR, Y = Distance)
    X = snrs_all;
    Y = distances;
    Z_th = caps_csi_perfect_all;
    Z_num = caps_csi_est_all;
    
    % Build surface grids - SWAPPED for proper orientation
    snr_unique = unique(X);
    d_unique = unique(Y);
    
    [Yg, Xg] = meshgrid(d_unique, snr_unique);  % SWAPPED order
    
    Zg_th = zeros(size(Xg));
    Zg_num = zeros(size(Xg));
    
    for k = 1:length(X)
        i = find(snr_unique == X(k));  % SWAPPED
        j = find(d_unique == Y(k));    % SWAPPED
        Zg_th(i, j) = Z_th(k);
        Zg_num(i, j) = Z_num(k);
    end
    
    % Check if numerical data exists
    has_numerical = any(Zg_num(:) > 0);
    
    % Create figure with IEEE formatting
    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 10, 7]);
    
    % Set paper size for IEEE format
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [10, 7]);
    
    % Create 3D axes
    ax = axes('Parent', fig);
    hold(ax, 'on');
    
    % Define colors matching matplotlib defaults
    color_perfect = [0.1216, 0.4667, 0.7059]; % #1f77b4
    color_est = [1.0000, 0.4980, 0.0549];     % #ff7f0e
    
    % Plot theoretical capacity surface
    surf1 = surf(ax, Yg, Xg, Zg_th, ...  % SWAPPED Xg and Yg
        'FaceColor', color_perfect, ...
        'FaceAlpha', 0.85, ...
        'EdgeColor', 'k', ...
        'LineWidth', 0.5);
    
    % Plot scatter points for theoretical
    scatter3(ax, Y, X, Z_th, 20, 'green', 'filled', ...  % SWAPPED X and Y
        'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    % Plot numerical capacity surface if exists
    if has_numerical
        surf2 = surf(ax, Yg, Xg, Zg_num, ...  % SWAPPED Xg and Yg
            'FaceColor', color_est, ...
            'FaceAlpha', 0.55, ...
            'EdgeColor', 'k', ...
            'LineWidth', 0.5);
        
        % Plot scatter points for numerical
        scatter3(ax, Y, X, Z_num, 15, color_est, 'filled');  % SWAPPED X and Y
    end
    
    % Axis formatting
    ax.XDir = 'reverse';  % Invert distance axis
    ax.YDir = 'normal';   % SNR from high to low (15 -> -20)
    
    xlim(ax, [min(Y) max(Y)]);  % SWAPPED
    ylim(ax, [min(X) max(X)]);  % SWAPPED
    
    xlabel(ax, 'UE Distance', 'FontName', 'Times New Roman', ...  % SWAPPED
        'FontSize', 11, 'Interpreter', 'latex');
    ylabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman', ...  % SWAPPED
        'FontSize', 11, 'Interpreter', 'latex');
    zlabel(ax, 'Capacity (bit/s/Hz)', 'FontName', 'Times New Roman', ...
        'FontSize', 11, 'Interpreter', 'latex');
    
    % Set view angle
    view(ax, -135, 25);
    
    % Grid settings
    grid(ax, 'on');
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.3;
    ax.GridColor = [0.15 0.15 0.15];
    
    % Font and line settings
    ax.FontName = 'Times New Roman';
    ax.FontSize = 10;
    ax.LineWidth = 1;
    ax.Box = 'on';
    ax.XColor = [0 0 0];
    ax.YColor = [0 0 0];
    ax.ZColor = [0 0 0];
    
    % Create custom legend
    % Create dummy patch objects for legend with matching colors
    h1 = patch(ax, 'XData', NaN, 'YData', NaN, 'ZData', NaN, ...
        'FaceColor', color_perfect, 'FaceAlpha', 0.85, ...
        'EdgeColor', 'k', 'LineWidth', 0.5, ...
        'DisplayName', 'Capacity of Perfect CSI');
    
    if has_numerical
        h2 = patch(ax, 'XData', NaN, 'YData', NaN, 'ZData', NaN, ...
            'FaceColor', color_est, 'FaceAlpha', 0.85, ...
            'EdgeColor', 'k', 'LineWidth', 0.5, ...
            'DisplayName', 'Capacity of Estimated CSI');
        legend_handles = [h1, h2];
    else
        legend_handles = h1;
    end
    
    leg = legend(ax, legend_handles, ...
        'Location', 'southoutside', ...
        'Orientation', 'horizontal', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 10, ...
        'NumColumns', 2);
    leg.Box = 'on';
    leg.EdgeColor = [0 0 0];
    leg.LineWidth = 0.5;
    leg.Color = [1 1 1];
    
    % Set all fonts to Times New Roman
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Times New Roman');
    
    % Configure for high-quality export
    set(fig, 'Renderer', 'painters');
    
    % Export to PDF
    filename = 'SNR_Cap_vs_Distance_v2';
    
    % Set figure properties for high-quality PDF export
    set(fig, 'Units', 'Inches');
    pos = get(fig, 'Position');
    set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
        'PaperSize', [pos(3), pos(4)]);
    
    print(fig, filename, '-dpdf', '-vector', '-r300', '-fillpage');
    
    fprintf('Figure saved as %s.pdf\n', filename);
end

function fig = plot_capacity_3d_lines(filepath)
    % Load the .mat file
    data = load(filepath);
    
    % Set default interpreter to LaTeX for all text
    set(groot, 'DefaultTextInterpreter', 'latex');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'DefaultLegendInterpreter', 'latex');
    set(groot, 'DefaultColorbarTickLabelInterpreter', 'latex');
    
    % Get all field names (excluding internal MATLAB fields)
    keys = fieldnames(data);
    keys = keys(~startsWith(keys, '__'));
    
    % Initialize arrays
    distances = [];
    snrs_all = [];
    caps_csi_perfect_all = [];
    caps_csi_est_all = [];
    
    % Extract data from each structure
    for i = 1:length(keys)
        entry = data.(keys{i});
        
        d = double(entry.ue_distance);
        snrs = double(entry.SNRs_Tx(:));
        
        caps_csi_perfect = double(entry.Numerical_Capacity_csi_v2(:));
        caps_csi_est = double(entry.Numerical_Capacity_est(:));
        
        distances = [distances; repmat(d, length(snrs), 1)];
        snrs_all = [snrs_all; snrs];
        caps_csi_perfect_all = [caps_csi_perfect_all; caps_csi_perfect];
        caps_csi_est_all = [caps_csi_est_all; caps_csi_est];
    end
    
    % Rename for clarity (X = SNR, Y = Distance)
    X = snrs_all;
    Y = distances;
    Z_th = caps_csi_perfect_all;
    Z_num = caps_csi_est_all;
    
    % Build surface grids - SWAPPED for proper orientation
    snr_unique = unique(X);
    d_unique = unique(Y);
    
    [Yg, Xg] = meshgrid(d_unique, snr_unique);  % SWAPPED order
    
    Zg_th = zeros(size(Xg));
    Zg_num = zeros(size(Xg));
    
    for k = 1:length(X)
        i = find(snr_unique == X(k));  % SWAPPED
        j = find(d_unique == Y(k));    % SWAPPED
        Zg_th(i, j) = Z_th(k);
        Zg_num(i, j) = Z_num(k);
    end
    
    % Check if numerical data exists
    has_numerical = any(Zg_num(:) > 0);
    
    % Create figure with IEEE formatting
    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 10, 7]);
    
    % Set paper size for IEEE format
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [10, 7]);
    
    % Create 3D axes
    ax = axes('Parent', fig);
    hold(ax, 'on');
    
    % Define colors matching matplotlib defaults
    color_perfect = [0.1216, 0.4667, 0.7059]; % #1f77b4
    color_est = [1.0000, 0.4980, 0.0549];     % #ff7f0e
    
    % Plot lines instead of surfaces
    % Plot lines for each SNR value (going across distances)
    for i = 1:length(snr_unique)
        % Plot perfect CSI capacity line
        plot3(ax, Yg(i,:), Xg(i,:), Zg_th(i,:), ...
            'Color', color_perfect, ...
            'LineStyle', '-', ...
            'LineWidth', 2, ...
            'Marker', 'o', ...
            'MarkerSize', 6, ...
            'MarkerFaceColor', 'green', ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.5);
        
        % Plot estimated CSI capacity line if exists
        if has_numerical
            plot3(ax, Yg(i,:), Xg(i,:), Zg_num(i,:), ...
                'Color', color_est, ...
                'LineStyle', '-', ...
                'LineWidth', 2, ...
                'Marker', 'o', ...
                'MarkerSize', 6, ...
                'MarkerFaceColor', color_est, ...
                'MarkerEdgeColor', 'k', ...
                'LineWidth', 0.5);
        end
    end
    
    % Axis formatting
    ax.XDir = 'reverse';  % Invert distance axis
    ax.YDir = 'normal';   % SNR from high to low (15 -> -20)
    
    xlim(ax, [min(Y) max(Y)]);  % SWAPPED
    ylim(ax, [min(X) max(X)]);  % SWAPPED
    
    xlabel(ax, 'UE Distance', 'FontName', 'Times New Roman', ...  % SWAPPED
        'FontSize', 11, 'Interpreter', 'latex');
    ylabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman', ...  % SWAPPED
        'FontSize', 11, 'Interpreter', 'latex');
    zlabel(ax, 'Capacity (bit/s/Hz)', 'FontName', 'Times New Roman', ...
        'FontSize', 11, 'Interpreter', 'latex');
    
    % Set view angle
    view(ax, -135, 25);
    
    % Grid settings
    grid(ax, 'on');
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.3;
    ax.GridColor = [0.15 0.15 0.15];
    
    % Font and line settings
    ax.FontName = 'Times New Roman';
    ax.FontSize = 10;
    ax.LineWidth = 1;
    ax.Box = 'on';
    ax.XColor = [0 0 0];
    ax.YColor = [0 0 0];
    ax.ZColor = [0 0 0];
    
    % Create custom legend
    % Create dummy patch objects for legend with matching colors
    h1 = plot3(ax, NaN, NaN, NaN, ...
        'Color', color_perfect, ...
        'LineStyle', '-', ...
        'LineWidth', 2, ...
        'Marker', 'o', ...
        'MarkerSize', 6, ...
        'MarkerFaceColor', 'green', ...
        'MarkerEdgeColor', 'k', ...
        'DisplayName', 'Capacity of Perfect CSI');
    
    if has_numerical
        h2 = plot3(ax, NaN, NaN, NaN, ...
            'Color', color_est, ...
            'LineStyle', '-', ...
            'LineWidth', 2, ...
            'Marker', 'o', ...
            'MarkerSize', 6, ...
            'MarkerFaceColor', color_est, ...
            'MarkerEdgeColor', 'k', ...
            'DisplayName', 'Capacity of Estimated CSI');
        legend_handles = [h1, h2];
    else
        legend_handles = h1;
    end
    
    leg = legend(ax, legend_handles, ...
        'Location', 'southoutside', ...
        'Orientation', 'horizontal', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 10, ...
        'NumColumns', 2);
    leg.Box = 'on';
    leg.EdgeColor = [0 0 0];
    leg.LineWidth = 0.5;
    leg.Color = [1 1 1];
    
    % Set all fonts to Times New Roman
    set(findall(fig, '-property', 'FontName'), 'FontName', 'Times New Roman');
    
    % Configure for high-quality export
    set(fig, 'Renderer', 'painters');
    
    % Export to PDF
    filename = 'SNR_Cap_vs_Distance_Lines';
    
    % Set figure properties for high-quality PDF export
    set(fig, 'Units', 'Inches');
    pos = get(fig, 'Position');
    set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
        'PaperSize', [pos(3), pos(4)]);
    
    print(fig, filename, '-dpdf', '-vector', '-r300', '-fillpage');
    
    fprintf('Figure saved as %s.pdf\n', filename);
end
function fig = plot_capacity_2d_lines(filepath)
    % Load the .mat file
    data = load(filepath);
    
    % Set default interpreter to LaTeX for all text
    set(groot, 'DefaultTextInterpreter', 'latex');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'DefaultLegendInterpreter', 'latex');
    set(groot, 'DefaultColorbarTickLabelInterpreter', 'latex');
    
    % Get all field names (excluding internal MATLAB fields)
    keys = fieldnames(data);
    keys = keys(~startsWith(keys, '__'));
    
    % Create figure with IEEE formatting
    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 8, 6]);
    
    % Set paper size for IEEE format
    set(fig, 'PaperPositionMode', 'auto');
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [8, 6]);
    
    % Create axes
    ax = axes('Parent', fig);
    hold(ax, 'on');
    
    % Define color map for different distances
    num_distances = length(keys);
    colors = lines(num_distances);
    
    % Arrays to store legend handles and labels
    legend_handles = [];
    legend_labels = {};
    
    % Plot each distance as a separate line
    for i = 1:length(keys)
        entry = data.(keys{i});
        
        distance = double(entry.ue_distance);
        snrs = double(entry.SNRs_Tx(:));
        caps_csi_perfect = double(entry.Numerical_Capacity_csi_v2(:));
        caps_csi_est = double(entry.Numerical_Capacity_est(:));
        
        % Sort by SNR for proper line plotting
        [snrs_sorted, idx] = sort(snrs);
        caps_perfect_sorted = caps_csi_perfect(idx);
        caps_est_sorted = caps_csi_est(idx);
        
        % Plot perfect CSI capacity
        h_perfect = plot(ax, snrs_sorted, caps_perfect_sorted, ...
            'Color', colors(i,:), ...
            'LineStyle', '-', ...
            'LineWidth', 1.5, ...
            'Marker', 'o', ...
            'MarkerSize', 6, ...
            'MarkerFaceColor', colors(i,:), ...
            'DisplayName', sprintf('Perfect CSI, d=%g m', distance));
        
        legend_handles = [legend_handles, h_perfect];
        legend_labels{end+1} = sprintf('Perfect CSI, d=%g m', distance);
        
        % Plot estimated CSI capacity if exists
        if any(caps_est_sorted > 0)
            h_est = plot(ax, snrs_sorted, caps_est_sorted, ...
                'Color', colors(i,:), ...
                'LineStyle', '--', ...
                'LineWidth', 1.5, ...
                'Marker', 's', ...
                'MarkerSize', 5, ...
                'MarkerFaceColor', 'none', ...
                'DisplayName', sprintf('Estimated CSI, d=%g m', distance));
            
            legend_handles = [legend_handles, h_est];
            legend_labels{end+1} = sprintf('Estimated CSI, d=%g m', distance);
        end
    end
    
    % Axis labels
    xlabel(ax, 'SNR (dB)', 'FontName', 'Times New Roman', ...
        'FontSize', 12, 'Interpreter', 'latex');
    ylabel(ax, 'Capacity (bit/s/Hz)', 'FontName', 'Times New Roman', ...
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
    filename = 'Capacity_vs_SNR_Lines';
    
    % Set figure properties for high-quality PDF export
    set(fig, 'Units', 'Inches');
    pos = get(fig, 'Position');
    set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', ...
        'PaperSize', [pos(3), pos(4)]);
    
    print(fig, filename, '-dpdf', '-vector', '-r300', '-fillpage');
    
    fprintf('Figure saved as %s.pdf\n', filename);
end


% Call the function with your file path
plot_capacity_3d_lines('results/simulation_results.mat');
plot_capacity_3d_surface('results/simulation_results.mat');
plot_capacity_2d_lines('results/simulation_results.mat');
