function [cluster_centroids, cluster_labels] = clusterDetections(estimated_ranges, estimated_velocities, eps, minPts)
    % CLUSTERDETECTIONS Clusters range and velocity detections and finds centroid points
    % Inputs:
    %   estimated_ranges: Array of estimated ranges (m)
    %   estimated_velocities: Array of estimated velocities (m/s)
    %   rng_grid: Range grid corresponding to detection_map rows
    %   dop_grid: Doppler (speed) grid corresponding to detection_map columns
    %   detection_map: Binary detection map (same as used for plotting)
    %   eps: Maximum distance for points to be in the same cluster (DBSCAN parameter)
    %   minPts: Minimum number of points to form a cluster (DBSCAN parameter)
    % Outputs:
    %   cluster_centroids: Matrix of [range, velocity] for each cluster centroid
    %   cluster_labels: Labels for each detection point (-1 for noise, 1, 2, ... for clusters)

    % Check if there are any detections
    if isempty(estimated_ranges) || isempty(estimated_velocities)
        cluster_centroids = [];
        cluster_labels = [];
        disp('No detections to cluster.');
        return;
    end

    % Combine ranges and velocities into a single matrix for clustering
    data = [estimated_ranges, estimated_velocities];

    % Normalize data for clustering (optional, but helps with DBSCAN)
    data_normalized = (data - mean(data)) ./ std(data);

    % Perform DBSCAN clustering
    % MATLAB's built-in dbscan requires the Statistics and Machine Learning Toolbox
    % If you don't have it, you can use a custom DBSCAN implementation
    cluster_labels = dbscan(data_normalized, eps, minPts);

    % If no clusters are found (all points are noise)
    if all(cluster_labels == -1)
        cluster_centroids = [];
        disp('No clusters found. All points are considered noise.');
        return;
    end

    % Find unique cluster labels (excluding noise, which is -1)
    unique_clusters = unique(cluster_labels(cluster_labels > 0));

    % Initialize cluster centroids
    cluster_centroids = zeros(length(unique_clusters), 2);

    % Compute centroid for each cluster
    for i = 1:length(unique_clusters)
        cluster_idx = (cluster_labels == unique_clusters(i));
        cluster_points = data(cluster_idx, :);
        cluster_centroids(i, :) = mean(cluster_points, 1); % [mean_range, mean_velocity]
    end
end