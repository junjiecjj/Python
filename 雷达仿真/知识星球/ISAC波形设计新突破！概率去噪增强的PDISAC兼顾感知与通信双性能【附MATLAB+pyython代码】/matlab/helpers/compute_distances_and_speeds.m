function [tar_distances, ue_distances, tar_speeds, ue_speeds] = ...
         compute_distances_and_speeds(radar_pos, tar_positions, ue_positions, v_tars_xyz, v_ues_xyz)

    % Number of time steps
    N = size(tar_positions, 2);

    % Initialize outputs
    tar_distances = zeros(1, N);
    ue_distances  = zeros(1, N);
    tar_speeds    = zeros(1, N);
    ue_speeds     = zeros(1, N);

    % Loop through time steps
    for k = 1:N
        % --- Distances (Euclidean norm)
        tar_distances(k) = norm(tar_positions(:,k) - radar_pos(:));
        ue_distances(k)  = norm(ue_positions(:,k) - radar_pos(:));

        % --- Speeds (signed radial, just take norm of v projected direction)
        tar_speeds(k) = norm(v_tars_xyz(:,k)) * sign(dot(v_tars_xyz(:,k), tar_positions(:,k) - radar_pos(:)));
        ue_speeds(k)  = norm(v_ues_xyz(:,k))  * sign(dot(v_ues_xyz(:,k),  ue_positions(:,k)  - radar_pos(:)));
    end
end