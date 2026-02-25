function plot_radar_vs_targets(target_locations, target_velocities, Rmax, opts)
    % plot_radar_vs_targets - Visualize radar and multiple targets in 2D
    %
    % Inputs:
    %   target_locations  - 3×N matrix, each column is a target position [x;y;z]
    %   target_velocities - 3×N matrix, each column is a target velocity [vx;vy;vz]
    %   Rmax             - Radar field of view (maximum detection range)
    %   opts             - Optional struct with fields:
    %                      .normalizeArrows (default: false)
    %                      .arrowFrac (default: 0.15)
    %                      .showFOV (default: true)

    set(groot, 'DefaultTextInterpreter', 'latex');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'DefaultLegendInterpreter', 'latex');
    set(groot, 'DefaultColorbarTickLabelInterpreter', 'latex');
    
    if nargin < 4, opts = struct(); end
    if ~isfield(opts, 'normalizeArrows'), opts.normalizeArrows = false; end
    if ~isfield(opts, 'arrowFrac'), opts.arrowFrac = 0.15; end
    if ~isfield(opts, 'showFOV'), opts.showFOV = true; end
    
    % Colors
    radarColor = [0 0 0];                % black
    tarColor   = [0.8500 0.3250 0.0980]; % orange/red
    fovColor   = [0 0.4470 0.7410];      % blue
    
    % Get number of targets
    num_targets = size(target_locations, 2);
    
    % Axis span for scaling
    allX = [target_locations(1,:), 0];
    allY = [target_locations(2,:), 0];
    spanX = max(allX) - min(allX);
    spanY = max(allY) - min(allY);
    axisSpan = max([spanX, spanY, Rmax]);
    
    % Compute scaling factor for velocity arrows (2D projection)
    vel_2d = target_velocities(1:2,:);
    targetSpeeds = vecnorm(vel_2d, 2, 1);
    maxSpeed = max([targetSpeeds, eps]);
    desiredMaxLen = opts.arrowFrac * axisSpan;
    
    if opts.normalizeArrows
        scale = desiredMaxLen;
        U_t = normalize2D(vel_2d) * scale;
    else
        scale = desiredMaxLen / maxSpeed;
        U_t = vel_2d * scale;
    end
    
    % === Plot ===
    figure; hold on; grid on; axis equal;
    xlabel('X [m]', 'FontWeight', 'bold', 'FontSize', 12);
    ylabel('Y [m]', 'FontWeight', 'bold', 'FontSize', 12);
    title(sprintf('Radar Field of View (R_{max} = %.0fm) with %d Target(s)', Rmax, num_targets), ...
        'FontWeight', 'bold', 'FontSize', 14);
    set(gca, 'FontSize', 11);
    
    % Plot radar detection circle (field of view)
    if opts.showFOV
        theta = linspace(0, 2*pi, 100);
        x_circle = Rmax * cos(theta);
        y_circle = Rmax * sin(theta);
        hFOV = fill(x_circle, y_circle, fovColor, 'FaceAlpha', 0.1, ...
            'EdgeColor', fovColor, 'LineWidth', 2, 'LineStyle', '--');
    end
    
    % Radar at origin
    hRadar = scatter(0, 0, 150, 's', 'MarkerFaceColor', radarColor, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    % Target positions (X-Y projection)
    Xt = target_locations(1,:);
    Yt = target_locations(2,:);
    
    % Range lines from radar to targets
    for i = 1:num_targets
        plot([0, Xt(i)], [0, Yt(i)], ':', ...
            'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    end
    
    % Velocity arrows
    hTarQuiv = quiver(Xt, Yt, U_t(1,:), U_t(2,:), 0, ...
        'Color', tarColor, 'LineWidth', 2.5, 'MaxHeadSize', 1.5);
    
    % Target markers on top of arrows
    markerSize = max(100, round(0.03*axisSpan^1.2));
    hTarMarkers = scatter(Xt, Yt, markerSize, 'o', ...
        'MarkerEdgeColor', tarColor, 'MarkerFaceColor', 'w', 'LineWidth', 1.8);
    
    % Labels for targets
    % labelOffset = 0.04 * axisSpan;
    % for i = 1:num_targets
    %     range_val = sqrt(target_locations(1,i)^2 + target_locations(2,i)^2);
    %     speed = norm(target_velocities(:, i));
    %     height = target_locations(3, i);
    %     text(Xt(i) + labelOffset, Yt(i) + labelOffset, ...
    %         sprintf('T%d: %.0fm, %.1fm/s\nZ=%.0fm', i, range_val, speed, height), ...
    %         'Color', tarColor, 'FontSize', 9, 'FontWeight', 'bold', ...
    %         'BackgroundColor', 'w', 'EdgeColor', tarColor, 'Margin', 2);
    % end
    
    % Radar label
    % text(labelOffset*1.5, -labelOffset*1.5, 'Radar', 'Color', radarColor, ...
    %     'FontWeight', 'bold', 'FontSize', 11, ...
    %     'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 2);
    
    % Axis limits
    margin = 0.15;
    lim = Rmax * (1 + margin);
    xlim([-lim, lim]);
    ylim([-lim, lim]);
    
    % Legend
    if opts.showFOV
        legend([hRadar, hFOV, hTarMarkers, hTarQuiv], ...
            {'Radar', 'Detection Range', 'Targets', 'Velocity Vectors'}, ...
            'Location', 'northeastoutside', 'FontSize', 10);
    else
        legend([hRadar, hTarMarkers, hTarQuiv], ...
            {'Radar', 'Targets', 'Velocity Vectors'}, ...
            'Location', 'northeastoutside', 'FontSize', 10);
    end
    
    box on;
    hold off;
end

% --- Helper to normalize 2D vectors ---
function UVn = normalize2D(UV)
    n = vecnorm(UV, 2, 1);
    UVn = UV ./ max(n, eps);
end