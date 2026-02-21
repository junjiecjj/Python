function plot_vary_crlb(numerical_inputs, mean_values, max_value, snrs, filename)
    [N, M] = size(numerical_inputs);
    K = length(mean_values);

    % Pre-allocate cell arrays to hold distances and cluster assignments
    distances   = cell(N,1);
    cluster_ids = cell(N,1);

    all_avg_d = zeros(K, N);  % [clusters x SNRs]

    % Step 1: Assign each point to the closest cluster and record the distance
    for n = 1:N
        seq = numerical_inputs(n, :);
        % Apply symmetric thresholding
        seq = seq((seq <= max_value) & (seq >= -max_value));
        P = numel(seq);

        d   = zeros(1, P);  % distances to assigned cluster
        ids = zeros(1, P);  % assigned cluster index

        for k = 1:P
            % Assign to cluster using closest mean
            [~, idx] = min(abs(seq(k) - mean_values));  % still works for negative values
            ids(k) = idx;
        
            % Use signed distance for plotting (not abs)
            d(k) = abs(seq(k) - mean_values(idx));  % preserves direction (can be negative)
        end

        distances{n}   = d;
        cluster_ids{n} = ids;
    end

    % Step 2: For each cluster, make a figure
    for c = 1:K
        figure; hold on;

        avg_d = nan(1, N);
        min_d = nan(1, N);
        max_d = nan(1, N);

        for n = 1:N
            d   = distances{n};
            ids = cluster_ids{n};
            sel = (ids == c);  % Only use points assigned to cluster c

            if any(sel)
                % Scatter plot all distances in cluster c for this sequence
                scatter(n * ones(1, sum(sel)), d(sel), 20, 'filled', ...
                    'MarkerFaceColor', [0.2 0.5 0.9], 'HandleVisibility', 'off');

                % Compute stats
                avg_d(n) = mean(d(sel));
                min_d(n) = min(d(sel));
                max_d(n) = max(d(sel));
            end
        end

        all_avg_d(c, :) = avg_d;

        % Plot lines
        plot(1:N, avg_d, '-o', 'LineWidth', 2, 'Color', 'k');
        plot(1:N, min_d, '+-', 'LineWidth', 1.5, 'Color', [0.3 0.3 0.3]);
        plot(1:N, max_d, '--', 'LineWidth', 1.5, 'Color', [0.3 0.3 0.3]);
        
        xticks(1:N);
        xticklabels(arrayfun(@num2str, snrs, 'UniformOutput', false));
        xlabel('SNR (dB)');
        ylabel('Distance to Assigned Cluster Mean');
        title(sprintf('Cluster %d: Assigned Point Distances and Summary Lines', c));
        legend('avg', 'min', 'max', 'Location', 'best');
        ylim([0, 1.05 * max(max_d(~isnan(max_d)))]);
        
        grid on;
    end

    % Save all avg_d values with filename containing max(mean_values)
    % max_val = max(mean_values);
    % max_str = strrep(sprintf('%.2f', max_val), '.', '_');  % e.g., 123.45 → '123_45'
    % filename = sprintf('./eval/avg_v_%s.mat', max_str);
    save(filename, 'all_avg_d', 'snrs');
end


