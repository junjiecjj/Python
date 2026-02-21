function [variances] = compute_var(numerical_inputs, num_of_targets, mean_values, max_value)
    % Get dimensions
    [N, M] = size(numerical_inputs);
    K = num_of_targets;

    % Compute cluster assignments for all rows at once
    mean_values_reshaped = reshape(mean_values, [1, 1, K]); % Shape: 1 x 1 x K
    numerical_inputs_reshaped = reshape(numerical_inputs, [N, M, 1]); % Shape: N x M x 1
    abs_diff = abs(numerical_inputs_reshaped - mean_values_reshaped); % Shape: N x M x K
    [~, cluster_idx] = min(abs_diff, [], 3); % Shape: N x M

    % Initialize output matrix
    variances = zeros(N, K);

    % Compute variances for each row
    for n = 1:N
        clusters = cluster_idx(n, :).'; % Cluster indices for row n, M x 1
        values = numerical_inputs(n, :).'; % Values in row n, M x 1

        % valid_mask = (values >= -max_value) & (values <= max_value);
        % values = values(valid_mask); % Shape: P x 1, P <= M
        % clusters = clusters(valid_mask); % Shape: P x 1, P <= M

        % Compute variances for each cluster
        vars_n = zeros(K, 1);
        for k = 1:K
            mask_k = (clusters == k);
            values_k = values(mask_k);
            if length(values_k) > 1
                % Compute the actual variance using the mean of values_k
                mean_k = mean(values_k);  % Use actual mean of the cluster data
                vars_n(k) = sum((values_k - mean_k).^2) / length(values_k);
            end
        end
        variances(n, :) = vars_n.'; % Assign to output row
    end
end