function targets = find_targets(data, num_of_targets)
    % Find target coordinates by identifying top N peaks and clustering detections.
    %
    % Parameters:
    % - data: matrix of shape (samples, 3) with [range, velocity, power]
    % - num_of_targets: number of top peaks to consider
    %
    % Returns:
    % - targets: matrix of shape (num_of_targets, 3) with mean [range, velocity, power] for each peak

    % Get indices of top N powers
    [~, sorted_indices] = sort(data(:,3), 'descend');
    
    num_of_targets = min(num_of_targets, length(sorted_indices));

    top_N_indices = sorted_indices(1:num_of_targets);
    peaks = data(top_N_indices, 1:2);

    % Compute Euclidean distances from all detections to all peaks
    detections_rv = data(:, 1:2);
    distances = sqrt(sum((reshape(detections_rv, size(detections_rv,1),1,2) ...
                - reshape(peaks, 1,num_of_targets,2)).^2, 3));

    % Compute standard deviation of distances to each peak
    std_dists = std(distances, 0, 1); % std along dimension 1 (for each peak)

    % Find the minimum distance to any peak for each detection
    [min_distances, closest_peaks] = min(distances, [], 2);

    % Filter detections: keep those where min distance <= std of the closest peak
    keep_mask = false(size(data, 1), 1);
    for i = 1:size(data, 1)
        closest_peak_idx = closest_peaks(i);
        if min_distances(i) <= std_dists(closest_peak_idx)
            keep_mask(i) = true;
        end
    end

    % Apply filter to data and distances
    filtered_data = data(keep_mask, :);
    filtered_distances = distances(keep_mask, :);

    % Assign each remaining detection to the nearest peak
    [~, assignments] = min(filtered_distances, [], 2);

    % Initialize output
    targets = nan(num_of_targets, 3);

    % Process each peak
    for peak_idx = 1:num_of_targets
        assigned_mask = (assignments == peak_idx);
        assigned_detections = filtered_data(assigned_mask, :);
        assigned_distances = filtered_distances(assigned_mask, peak_idx);

        if isempty(assigned_detections)
            continue; % Leave as NaN
        end

        % Compute standard deviation of distances for this peak (post-filtering)
        std_dist = std(assigned_distances);

        % Filter detections within 1 * std for this peak
        close_mask = assigned_distances <= std_dist;
        close_detections = assigned_detections(close_mask, :);

        if isempty(close_detections)
            continue; % Leave as NaN
        end

        % Compute mean coordinates [range, velocity, power]
        targets(peak_idx, :) = mean(close_detections, 1);
    end
end


