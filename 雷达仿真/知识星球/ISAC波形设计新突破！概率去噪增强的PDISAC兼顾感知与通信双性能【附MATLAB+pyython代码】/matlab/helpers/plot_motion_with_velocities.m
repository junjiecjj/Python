function plot_motion_with_velocities(radar_pos, tar_positions, ue_positions, tar_vels, ue_vels, time_steps, opts)


    if nargin < 7, opts = struct(); end
    if ~isfield(opts,'normalizeArrows'), opts.normalizeArrows = false; end
    if ~isfield(opts,'arrowFrac'),        opts.arrowFrac = 0.1; end
    if ~isfield(opts,'markerOffset'),     opts.markerOffset = 0.04; end
    if ~isfield(opts, 'labelOffset'), opts.labelOffset = 3.5; end

    % Colors
    tarColor   = [0.8500 0.3250 0.0980]; % orange/red
    ueColor    = [0 0.4470 0.7410];      % blue
    radarColor = [0 0 0];                % black

    % Extract XY
    Xt = tar_positions(1,:); Yt = tar_positions(2,:);
    Xu = ue_positions(1,:);  Yu = ue_positions(2,:);
    Rt = radar_pos(:);

    % Axis span for scaling
    allX = [Xt, Xu, Rt(1)]; allY = [Yt, Yu, Rt(2)];
    spanX = range(allX); spanY = range(allY);
    axisSpan = max(max(spanX, spanY), 1);

    % Compute scaling factor
    tarSpeeds2 = vecnorm(tar_vels(1:2,:),2,1);
    ueSpeeds2  = vecnorm(ue_vels(1:2,:),2,1);
    maxSpeed = max([tarSpeeds2, ueSpeeds2, eps]);

    desiredMaxLen = opts.arrowFrac * axisSpan;
    if opts.normalizeArrows
        scale = desiredMaxLen;
        U_t = normalize2D(tar_vels(1:2,:)) * scale;
        U_u = normalize2D(ue_vels(1:2,:)) * scale;
    else
        scale = desiredMaxLen / maxSpeed;
        U_t = tar_vels(1:2,:) * scale;
        U_u = ue_vels(1:2,:) * scale;
    end

    % === Plot ===
    figure; hold on; grid on; axis equal;
    xlabel('X [m]'); ylabel('Y [m]');
    title('Target & UE Motion with Velocity Directions','FontWeight','bold','FontSize',14);
    set(gca,'FontSize',11);

    % Radar
    hRadar = scatter(Rt(1), Rt(2), 120, 's', 'MarkerFaceColor', radarColor, 'MarkerEdgeColor', 'k');

    % Target Path + Arrows
    hTarPath = plot(Xt, Yt, '-', 'Color', tarColor, 'LineWidth', 1.8);
    hTarQuiv = quiver(Xt, Yt, U_t(1,:), U_t(2,:), 0, ...
        'Color', tarColor, 'LineWidth', 2, 'MaxHeadSize', 2);

    % UE Path + Arrows
    hUePath = plot(Xu, Yu, '-', 'Color', ueColor, 'LineWidth', 1.8);
    hUeQuiv = quiver(Xu, Yu, U_u(1,:), U_u(2,:), 0, ...
        'Color', ueColor, 'LineWidth', 2, 'MaxHeadSize', 2);

    % Markers on top of arrows (smaller size than before)
    markerSize = max(25, round(0.015*axisSpan^1.1));
    scatter(Xt, Yt, markerSize, 'o', 'MarkerEdgeColor', tarColor, 'MarkerFaceColor', 'w', 'LineWidth',1.2);
    scatter(Xu, Yu, markerSize, 'o', 'MarkerEdgeColor', ueColor, 'MarkerFaceColor', 'w', 'LineWidth',1.2);

    % labelOffset = opts.labelOffset;
    % % Labels (adjust UE labels to offset vertically)
    % for k = 1:length(Xt)
    %     text(Xt(k) + labelOffset, Yt(k) + labelOffset, ...
    %         sprintf('T@%.1fs', time_steps(k)), 'Color', tarColor, ...
    %         'FontSize', 5, 'FontWeight', 'bold');
    % end
    % for k = 1:length(Xu)
    %     text(Xu(k), Yu(k) + 2*labelOffset, ...  % Vertical offset only, doubled for clarity
    %         sprintf('U@%.1fs', time_steps(k)), 'Color', ueColor, ...
    %         'FontSize', 5, 'FontWeight', 'bold');
    % end
    % text(Rt(1) + labelOffset, Rt(2), 'Radar', 'Color', radarColor, 'FontWeight','bold');

    % Axis limits
    margin = 1;
    xlim([min(allX)-margin*axisSpan, max(allX)+margin*axisSpan]);
    ylim([min(allY)-margin*axisSpan, max(allY)+margin*axisSpan]);

    % Legend
    legend([hRadar, hTarPath, hTarQuiv, hUePath, hUeQuiv], ...
        {'Radar','Target Path','Target Velocities','UE Path','UE Velocities'}, ...
        'Location','northeastoutside');

    box on;
end


% --- Helper to normalize 2D vectors ---
function UVn = normalize2D(UV)
    n = vecnorm(UV,2,1);
    UVn = UV ./ max(n,eps);
end
