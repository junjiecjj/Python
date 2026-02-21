function [est_ranges, est_velocities, detected_powers] = applyCFAR2D(resp, rng_grid, dop_grid, cfar2D, num_guard, num_train, vrelmax, Rmax)

    % Prepare the detection matrix
    rd_map = abs(resp).^2;

    % Define valid CUT indices that fit within the matrix
    [n_range, n_doppler] = size(rd_map);
    border_size = num_guard + num_train; % Total cells needed on each side

    % Restrict indices to avoid edge effects
    range_start = border_size + 1;
    range_end = n_range - border_size;
    doppler_start = border_size + 1;
    doppler_end = n_doppler - border_size;

    [range_idx, doppler_idx] = meshgrid(range_start:range_end, doppler_start:doppler_end);
    cut_idx = [range_idx(:)'; doppler_idx(:)'];

    % Apply CFAR detection
    detections = cfar2D(rd_map, cut_idx);

    % Create full detection map (initialize with zeros)
    detection_map = zeros(size(rd_map));
    % Place detections at their corresponding indices
    valid_idx = sub2ind(size(rd_map), range_idx(:), doppler_idx(:));
    detection_map(valid_idx) = detections;

    % Get all detected target locations
    [det_range_idx, det_dop_idx] = find(detection_map);
    detected_ranges = rng_grid(det_range_idx);
    detected_speeds = dop_grid(det_dop_idx);

    % Filter detections within specified range [0, 200] m and velocity [-60, 60] m/s
    range_mask = (detected_ranges >= 0) & (detected_ranges <= Rmax);
    speed_mask = (detected_speeds >= -vrelmax) & (detected_speeds <= vrelmax);
    valid_mask = range_mask & speed_mask;

    % Apply filter to get only targets within specified region
    filtered_range_idx = det_range_idx(valid_mask);
    filtered_dop_idx = det_dop_idx(valid_mask);
    est_ranges = detected_ranges(valid_mask);
    est_velocities = detected_speeds(valid_mask);

    % Get the cell power values at the filtered detections
    num_detections = length(filtered_range_idx);
    detected_powers = zeros(num_detections, 1);
    for i = 1:num_detections
        detected_powers(i) = rd_map(filtered_range_idx(i), filtered_dop_idx(i));
    end
end