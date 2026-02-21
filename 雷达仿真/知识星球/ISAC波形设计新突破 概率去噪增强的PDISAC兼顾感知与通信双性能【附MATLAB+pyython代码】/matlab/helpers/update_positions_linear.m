function [radar_pos, tar_positions, ue_positions, v_tar_all, v_ue_all] = ...
    update_positions_linear(l_radar, l_tar, l_ue, time_steps, v_tar_range, v_ue_range)

    % Inputs:
    % l_radar    : [3x1] radar location (fixed)
    % l_tar      : [3x1] initial target position
    % l_ue       : [3x1] initial UE position
    % time_steps : [1xN] time instants (seconds)
    % v_tar_range: [v_start, v_end] target radial speed (m/s)
    % v_ue_range : [v_start, v_end] UE radial speed (m/s)
    %
    % Outputs:
    % radar_pos    : [3x1] radar position
    % tar_positions: [3xN] target positions
    % ue_positions : [3xN] UE positions
    % v_tar_all    : [3xN] target velocity vectors
    % v_ue_all     : [3xN] UE velocity vectors

    N = length(time_steps);
    tar_positions = zeros(3,N);
    ue_positions  = zeros(3,N);
    v_tar_all     = zeros(3,N);
    v_ue_all      = zeros(3,N);

    radar_pos = l_radar;

    % Unit directions from radar to initial positions
    dir_tar = (l_tar - l_radar) / norm(l_tar - l_radar);
    dir_ue  = (l_ue - l_radar) / norm(l_ue - l_radar);

    % Linearly varying signed speeds
    speeds_tar = linspace(v_tar_range(1), v_tar_range(2), N);
    speeds_ue  = linspace(v_ue_range(1), v_ue_range(2), N);

    % Loop over steps
    prev_time = 0;
    pos_tar = l_tar;
    pos_ue  = l_ue;

    for k = 1:N
        dt = time_steps(k) - prev_time;
        prev_time = time_steps(k);

        % Radial velocity vectors (sign determines toward/away)
        v_tar_k = speeds_tar(k) * dir_tar;
        v_ue_k  = speeds_ue(k)  * dir_ue;

        % Update positions
        pos_tar = pos_tar + v_tar_k * dt;
        pos_ue  = pos_ue  + v_ue_k * dt;

        % Store
        tar_positions(:,k) = pos_tar;
        ue_positions(:,k)  = pos_ue;
        v_tar_all(:,k)     = v_tar_k;
        v_ue_all(:,k)      = v_ue_k;
    end
end