function [target_coords, target_powers] = kmeans_target_detection(data, number_of_target)

    det_ranges = data(:,1);
    det_velocities = data(:,2);
    detected_powers = data(:,3);
    
   
    % Reshape inputs to vectors
    ranges = det_ranges(:);
    velocities = det_velocities(:);
    powers = detected_powers(:);
   
    
    if length(ranges) <= number_of_target
        valid_idx = powers >= 0;
        number_of_target = length(ranges);
    else
        median_power = median(powers, 'all');
        valid_idx = powers >= median_power;
    end
    

    % Create coordinate matrix for valid points
    coords = [ranges(valid_idx), velocities(valid_idx)];
    valid_powers = powers(valid_idx);
    
    % Perform K-means clustering
    [idx, centroids] = kmeans(coords, number_of_target, ...
        'Distance', 'sqeuclidean', ...
        'MaxIter', 10, ...
        'Replicates', 5);
    
    % Merge close centroids
    min_distance = 0.1; % Adjust this threshold as needed
    merged = true;
    while merged
        merged = false;
        for i = 1:size(centroids, 1)
            for j = i+1:size(centroids, 1)
                if i ~= j && ~isempty(centroids(i,:)) && ~isempty(centroids(j,:))
                    dist = sqrt(sum((centroids(i,:) - centroids(j,:)).^2));
                    if dist < min_distance
                        % Merge by taking mean
                        centroids(i,:) = mean([centroids(i,:); centroids(j,:)]);
                        centroids(j,:) = [];
                        % Update cluster assignments
                        idx(idx == j) = i;
                        merged = true;
                    end
                end
            end
        end
        % Remove empty centroids
        centroids = centroids(~cellfun(@isempty, mat2cell(centroids, ones(size(centroids,1),1))), :);
    end
    
    % Calculate final target coordinates and powers
    target_coords = centroids;
    target_powers = zeros(size(centroids, 1), 1);
    
    for i = 1:size(centroids, 1)
        cluster_idx = idx == i;
        if any(cluster_idx)
            target_powers(i) = mean(valid_powers(cluster_idx));
        end
    end
    
    % Ensure output dimensions
    if isempty(target_coords)
        target_coords = zeros(0, 2);
        target_powers = zeros(0, 1);
    end
end