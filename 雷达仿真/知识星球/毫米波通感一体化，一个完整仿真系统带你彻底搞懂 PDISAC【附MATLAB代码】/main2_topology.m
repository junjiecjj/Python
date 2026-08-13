% TOPOLOGY  (consolidated Group 2: main2_topology)
% -------------------------------------------------------------------------
% Builds the PDISAC scene from the configuration and (optionally) renders the
% 2-D plan view.
%
% Run as a script (shares the base workspace, as before):
%     main1_matlab_config;   % or let the build cell call it
%     main2_topology;
%
% Shared kernel helpers kept as standalone files (unchanged, used elsewhere):
%     func_generate_static_scatterers.m, func_get_array_response.m
%
% To build the scene WITHOUT drawing the figure, set  TOPO_NO_PLOT = true;
% before running (used by batch/statistics drivers).
%
% Populates: N_ue, UE_position, UE_vel, UE_rcs, UE_motion,
%   N_tars, Tars_position, Tars_vel, Tars_rcs, Tars_motion,
%   N_scats, Scats_position, Scats_vel, Scats_rcs, Scats_motion.

%% Build scene ---------------------------------------------------------------
main1_matlab_config;

% ---- User equipment (UE) ----
N_ue = 1;
UE_position = PDISAC_cfg.scene.ue_position_m(:);
UE_vel = PDISAC_cfg.scene.ue_velocity_mps(:);
UE_motion = phased.Platform('InitialPosition', UE_position, 'Velocity', UE_vel);

target_rcs_min_db = 10 * log10(PDISAC_cfg.scene.target_rcs_range(1));
target_rcs_max_db = 10 * log10(PDISAC_cfg.scene.target_rcs_range(2));
UE_rcs_db = target_rcs_min_db + ...
    (target_rcs_max_db - target_rcs_min_db) * randn(1, N_ue);
UE_rcs = 10.^(UE_rcs_db / 10);

% ---- Moving targets ----
Tars_position = PDISAC_cfg.scene.target_positions_m;
Tars_vel = PDISAC_cfg.scene.target_velocities_mps;
N_tars = size(Tars_position, 2);
Tars_rcs_db = target_rcs_min_db + ...
    (target_rcs_max_db - target_rcs_min_db) * randn(1, N_tars);
Tars_rcs = 10.^(Tars_rcs_db / 10);

% Append the UE to the target set (UE also acts as a reflector for sensing).
Tars_position = [Tars_position, UE_position];
Tars_vel = [Tars_vel, UE_vel];
Tars_rcs = [Tars_rcs, UE_rcs];
N_tars = size(Tars_position, 2);
Tars_motion = phased.Platform( ...
    'InitialPosition', Tars_position, 'Velocity', Tars_vel);

% ---- stationary scatterers ----
N_scats = PDISAC_cfg.scene.scatterers;
Scats_position = func_generate_static_scatterers( ...
    N_scats, Region_of_interest);
Scats_vel = zeros(size(Scats_position));
scatter_rcs_min_db = 10 * log10(PDISAC_cfg.scene.scatterer_rcs_range(1));
scatter_rcs_max_db = 10 * log10(PDISAC_cfg.scene.scatterer_rcs_range(2));
Scats_rcs_db = scatter_rcs_min_db + ...
    (scatter_rcs_max_db - scatter_rcs_min_db) * randn(1, N_scats);
Scats_rcs = 10.^(Scats_rcs_db / 10);
Scats_motion = phased.Platform( ...
    'InitialPosition', Scats_position, 'Velocity', Scats_vel);

%% Visualize  (former plot_system_topo.m) ------------------------------------
if ~exist('TOPO_NO_PLOT', 'var') || ~TOPO_NO_PLOT
    show_topology(Tx_position, Rx_position, UE_position, UE_vel, ...
        Tars_position, Tars_vel, Scats_position, ...
        N_tx_ant, N_rx_ant, N_ue_ant, N_tars, N_scats, N_ue, Lambda, N_ref_tars);
end


% =========================================================================
%  LOCAL FUNCTIONS  (former plot_system_topo.m)
% =========================================================================
function show_topology(Tx_position, Rx_position, UE_position, UE_vel, ...
        Tars_position, Tars_vel, Scats_position, ...
        N_tx_ant, N_rx_ant, N_ue_ant, N_tars, N_scats, N_ue, Lambda, N_ref_tars)
% Angle bookkeeping, console report, and 2-D plan-view rendering.

    % ---- Moving targets (Tx -> target -> Rx) --------------------------------
    Rel_positions = Tars_position - Tx_position;
    [Azi_Tx_tars, Ele_Tx_tars, Range_tars] = ...
        cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_tars = rad2deg(Azi_Tx_tars);
    Ele_Tx_tars = rad2deg(Ele_Tx_tars);
    Azi_tars_Rx = Azi_Tx_tars;
    Ele_tars_Rx = Ele_Tx_tars;

    for i = 1:N_tars
        fprintf('Moving Target %d: Range = %.2f, Azimuth = %.2f deg, Elevation = %.2f deg\n', ...
            i, Range_tars(i), Azi_Tx_tars(i), Ele_Tx_tars(i));
    end

    % ---- stationary scatterers --------------------------------------------------
    Rel_positions = Scats_position - Tx_position;
    [Azi_Tx_scats, Ele_Tx_scats, Range_scats] = ...
        cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_scats = rad2deg(Azi_Tx_scats);
    Ele_Tx_scats = rad2deg(Ele_Tx_scats);

    for i = 1:N_scats
        fprintf('Static Target %d: Range = %.2f m, Azimuth = %.2f deg, Elevation = %.2f deg\n', ...
            i, Range_scats(i), Azi_Tx_scats(i), Ele_Tx_scats(i));
        if i == 10, break; end
    end

    % ---- UE -----------------------------------------------------------------
    Rel_positions = UE_position - Tx_position;
    [Azi_Tx_UE, Ele_Tx_UE, Range_Tx_UE] = ...
        cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_UE = rad2deg(Azi_Tx_UE);
    Ele_Tx_UE = rad2deg(Ele_Tx_UE);

    Rel_positions = UE_position - Tx_position;
    R_hat = Rel_positions / norm(Rel_positions);
    Radial_UE_vel = dot(UE_vel, R_hat);

    for i = 1:N_ue
        fprintf('UE %d: Range = %.2f, Azimuth = %.2f deg, Elevation = %.2f deg, Radial Velocity = %.2f m/s\n', ...
            i, Range_Tx_UE(i), Azi_Tx_UE(i), Ele_Tx_UE(i), Radial_UE_vel(i));
    end

    plot_fov(Tx_position, Rx_position, UE_position, UE_vel, ...
        Tars_position, Tars_vel, Scats_position, ...
        N_tx_ant, N_rx_ant, N_ue_ant, Lambda, ...
        Azi_Tx_tars, 90-Ele_Tx_tars, ...
        Azi_tars_Rx, 90-Ele_tars_Rx, ...
        Azi_Tx_UE,   90-Ele_Tx_UE, ...
        Azi_Tx_UE,   90-Ele_Tx_UE, ...
        N_ref_tars);
    view(2)   % top-down plan view
end

function plot_fov(Tx_position, Rx_position, UE_position, UE_vel, ...
    Tars_position, Tars_vel, Scats_position, ...
    N_tx_ant, N_rx_ant, N_ue_ant, Lambda, ...
    Azi_Tx_tars, Ele_Tx_tars, ...
    Azi_tars_Rx, Ele_tars_Rx, ...
    Azi_Tx_UE,   Ele_Tx_UE, ...
    Azi_UE,      Ele_UE, ...
    N_ref_tars)
%PLOT_FOV  Publication-quality 2-D plan view of the ISAC scenario.

    if nargin < 20 || isempty(N_ref_tars)
        N_ref_tars = 6;   % draw the closest NLoS reflectors by default
    end

    fontName = 'Times New Roman';

    % -- Palette --------------------------------------------------------
    c_bs   = [0.15 0.15 0.15];
    c_ue   = [0.00 0.45 0.74];
    c_tar  = [0.85 0.20 0.20];
    c_scat = [0.55 0.55 0.55];
    c_nlos = [0.20 0.62 0.30];
    c_sen  = [0.93 0.55 0.13];

    % -- Drop the UE copy that config appends as the last "target" ------
    is_ue = vecnorm(Tars_position - UE_position) < 1e-6;
    Tp = Tars_position(:, ~is_ue);
    Tv = Tars_vel(:,      ~is_ue);
    nT = size(Tp, 2);

    bs = Tx_position(:);
    ue = UE_position(:);

    % -- Figure -----------------------------------------------------------
    figure('Color','w','Name','ISAC System Topology','Position',[100 100 1300 650]);
    ax = axes; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    set(ax,'Layer','top','FontSize',11,'GridAlpha',0.12,'FontName',fontName, ...
           'TickLabelInterpreter','latex');

    % -- Region of interest -------------------------------------------
    roi_x = [0 100];  roi_y = [-100 100];
    plot([roi_y(1) roi_y(1) roi_y(2) roi_y(2) roi_y(1)], ...
         [roi_x(1) roi_x(2) roi_x(2) roi_x(1) roi_x(1)], ...
         ':', 'Color',[0.6 0.6 0.6], 'LineWidth',1.0, ...
         'DisplayName','Region of interest');

    % -- Range rings from the BS ------------------------------------------
    Rmax = max([vecnorm(Tp - bs), norm(ue - bs)]);
    ring_step = 25;
    ring_r = ring_step:ring_step:ceil(Rmax/ring_step)*ring_step;
    th = linspace(0, 2*pi, 200);
    for k = 1:numel(ring_r)
        r = ring_r(k);
        hv = 'off'; if k==1, hv='on'; end
        plot(bs(2)+r*cos(th), bs(1)+r*sin(th), '-', ...
             'Color',[0.85 0.85 0.85], 'LineWidth',0.6, ...
             'HandleVisibility',hv, 'DisplayName','Range rings');
        text(bs(2)+r*cosd(8), bs(1)+r*sind(8), sprintf('%d m',r), ...
             'Color',[0.6 0.6 0.6], 'FontSize',8, 'FontName',fontName, ...
             'Interpreter','latex', 'Clipping','on');
    end

    % -- NLoS reflectors ---------------------------------------------------
    All_tars_position = [Tp, Scats_position];
    Rel_positions      = UE_position - All_tars_position;
    Range_All_tars     = sqrt(sum(Rel_positions.^2, 1));
    [~, idx]           = sort(Range_All_tars, 'ascend');
    closest_idx        = idx(1:min(N_ref_tars, numel(idx)));
    Ref_tars_position  = All_tars_position(:, closest_idx);

    % -- Sensing links: BS <-> target ---------------------------------------
    for i = 1:nT
        hv = 'off'; if i==1, hv='on'; end
        plot([bs(2) Tp(2,i)], [bs(1) Tp(1,i)], '-', ...
             'Color',[c_sen 0.55], 'LineWidth',1.3, ...
             'HandleVisibility',hv, ...
             'DisplayName','Sensing link (Tx $\rightarrow$ target $\rightarrow$ Rx)');
    end

    % -- Sensing (clutter) links: BS <-> static scatterer -------------------
    nS = size(Scats_position, 2);
    for i = 1:nS
        hv = 'off'; if i==1, hv='on'; end
        plot([bs(2) Scats_position(2,i)], [bs(1) Scats_position(1,i)], '-', ...
             'Color',[c_scat 0.08], 'LineWidth',0.6, ...
             'HandleVisibility',hv, ...
             'DisplayName','Sensing link (Tx $\rightarrow$ scatterer $\rightarrow$ Rx)');
    end

    % -- Communication LoS link: BS -> UE -------------------------------------
    plot([bs(2) ue(2)], [bs(1) ue(1)], '-', ...
         'Color',c_ue, 'LineWidth',2.2, ...
         'DisplayName','Comm LoS (Tx $\rightarrow$ UE)');

    % -- Communication NLoS links: BS -> reflector -> UE -----------------------
    for i = 1:size(Ref_tars_position, 2)
        hv = 'off'; if i==1, hv='on'; end
        refl = Ref_tars_position(:,i);
        plot([bs(2) refl(2) ue(2)], [bs(1) refl(1) ue(1)], '--', ...
             'Color',c_nlos, 'LineWidth',1.8, ...
             'HandleVisibility',hv, ...
             'DisplayName','Comm NLoS (Tx $\rightarrow$ reflector $\rightarrow$ UE)');
        scatter(refl(2), refl(1), 70, c_nlos, 'o', 'LineWidth',1.6, ...
                'HandleVisibility','off');
    end

    % -- stationary scatterers --------------------------------------------------
    scatter(Scats_position(2,:), Scats_position(1,:), 26, c_scat, 'o', ...
            'filled', 'MarkerFaceAlpha',0.45, 'DisplayName','stationary scatterers');

    % -- Radar targets + velocity vectors ------------------------------------
    scatter(Tp(2,:), Tp(1,:), 95, c_tar, '^', 'filled', ...
            'MarkerEdgeColor','k', 'DisplayName','Moving Targets');
    for i = 1:nT
        quiver(Tp(2,i), Tp(1,i), Tv(2,i), Tv(1,i), 0, ...
               'Color',c_tar, 'LineWidth',2.0, 'MaxHeadSize',0.8, ...
               'HandleVisibility','off');
        text(Tp(2,i)+2.0, Tp(1,i)+2.0, ...
             sprintf('Tar ${%d}$\\,(%.0f m/s)', i, norm(Tv(:,i))), ...
             'Interpreter','latex', 'FontSize',10, 'FontName',fontName, ...
             'Color',c_tar);
    end

    % -- UE + velocity vector -------------------------------------------------
    scatter(ue(2), ue(1), 150, c_ue, 's', 'filled', ...
            'MarkerEdgeColor','k', 'DisplayName','UE (Rx)');
    quiver(ue(2), ue(1), UE_vel(2), UE_vel(1), 0, ...
           'Color',c_ue, 'LineWidth',1.4, 'MaxHeadSize',1.0, ...
           'HandleVisibility','off');
    text(ue(2)+2.5, ue(1)+2.5, sprintf('UE (%.0f m/s)', norm(UE_vel)), ...
         'Interpreter','latex', 'FontSize',10, 'FontWeight','bold', ...
         'FontName',fontName, 'Color',c_ue);

    % -- Base station (co-located Tx/Rx) ---------------------------------------
    scatter(bs(2), bs(1), 220, c_bs, 'p', 'filled', ...
            'MarkerEdgeColor','k', 'DisplayName','BS (Tx/Rx)');
    text(bs(2)+2.5, bs(1)-4.0, 'BS (Tx/Rx)', ...
         'Interpreter','latex', 'FontSize',11, 'FontWeight','bold', ...
         'FontName',fontName, 'Color',c_bs);

    % -- Cosmetics ------------------------------------------------------------
    pad_y = 8;
    pad_x = 2 * pad_y;

    axis equal;
    xlim([roi_y(1)-pad_x, roi_y(2)+pad_x]);
    ylim([roi_x(1)-pad_y, roi_x(2)+pad_y]);

    xlabel('$y$ (m)', 'Interpreter','latex', 'FontSize',12, 'FontName',fontName);
    ylabel('$x$ (m)', 'Interpreter','latex', 'FontSize',12, 'FontName',fontName);
    title('ISAC System Topology', 'Interpreter','latex', 'FontSize',13, ...
          'FontName',fontName);
    lgd = legend('Location','northeastoutside', 'FontSize',9.5, 'Box','on', ...
                 'Interpreter','latex');
    lgd.FontName = fontName;
    hold(ax,'off');
end
