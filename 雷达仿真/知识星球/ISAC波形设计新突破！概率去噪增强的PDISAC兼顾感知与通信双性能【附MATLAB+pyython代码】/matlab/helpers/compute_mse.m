function [mse_matrix] = compute_mse(numerical_inputs, num_of_targets, mean_values, max_value)
    % Get dimensions
    [N, M] = size(numerical_inputs);
    K = num_of_targets;

    % Compute cluster assignments for all rows at once
    mean_values_reshaped = reshape(mean_values, [1, 1, K]);            % 1 x 1 x K
    numerical_inputs_reshaped = reshape(numerical_inputs, [N, M, 1]);  % N x M x 1
    abs_diff = abs(numerical_inputs_reshaped - mean_values_reshaped);  % N x M x K
    [~, cluster_idx] = min(abs_diff, [], 3);                           % N x M

    % Initialize output matrix
    mse_matrix = zeros(N, K);

    % Compute MSE for each row
    for n = 1:N
        clusters = cluster_idx(n, :).';              % Cluster indices for row n, M x 1
        values = numerical_inputs(n, :).';           % Values in row n, M x 1

        % Apply validity mask
        % valid_mask = (values >= -max_value) & (values <= max_value);
        % values = values(valid_mask);
        % clusters = clusters(valid_mask);

        % Compute MSE for each cluster
        mse_n = zeros(K, 1);
        for k = 1:K
            mask_k = (clusters == k);
            values_k = values(mask_k);

            if ~isempty(values_k)
                % mean squared error relative to cluster mean
                mse_n(k) = mean((values_k - mean_values(k)).^2);
            end
        end

        mse_matrix(n, :) = mse_n.';  % Assign to output row
    end
end



