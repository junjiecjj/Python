function [hat_r_tars, hat_v_tars] = est_rv_seq_z(z_time, T_prbs, num_of_targets, fs, lambda)

    hat_r_tars = zeros(num_of_targets, 1);
    hat_v_tars = zeros(num_of_targets, 1);

    % Get dimensions
    [~, N_sym] = size(z_time);
    fasttime = unigrid(0, 1/fs, T_prbs, '[)');
    rangebins = (physconst('Lightspeed') * fasttime / 2);

    %% Range Estimation: r_hat = argmax_r { sum_n |y(r,n)|^2 }
    sum_z_time = sum(abs(z_time), 2);
    [~, range_detect] = findpeaks(pulsint(sum_z_time, 'noncoherent'), 'SortStr', 'descend');

    try
        best_peak = range_detect(1:num_of_targets);
    catch
        best_peak = zeros(num_of_targets, 1);
    end

    for k = 1:num_of_targets
        if best_peak(k) > 0
            idx = best_peak(k);
            ts = squeeze(z_time(idx, :)).';

            % Doppler estimation: f_d_hat = argmax_fd { |FFT{y(r_hat,n)}|^2 }
            [Pxx, F] = periodogram(ts, [], N_sym, 1/(2*T_prbs), 'centered');
            [~, max_idx] = max(Pxx);
            doppler_freq = F(max_idx);

            % Velocity conversion
            hat_v_tars(k) = dop2speed(-doppler_freq / 2, lambda);
            hat_r_tars(k) = rangebins(idx);
        else
            hat_r_tars(k) = 0;
            hat_v_tars(k) = 0;
        end
    end

    % %% Doppler Estimation and Plotting 
    % fig_doppler = figure('Position', [150, 150, 500, 500]);
    % set(fig_doppler, 'Color', 'w');
    % 
    % for k = 1:num_of_targets
    %     if best_peak(k) > 0
    %         idx = best_peak(k);
    %         ts = squeeze(z_time(idx, :)).';
    % 
    %         % Doppler estimation: f_d_hat = argmax_fd { |FFT{y(r_hat,n)}|^2 }
    %         [Pxx, F] = periodogram(ts, [], N_sym, 1/(2*T_prbs), 'centered');
    %         [~, max_idx] = max(Pxx);
    %         doppler_freq = F(max_idx);
    % 
    %         % Velocity conversion: v = -lambda * f_d / 2
    %         hat_v_tars(k) = dop2speed(-doppler_freq / 2, lambda);
    %         hat_r_tars(k) = rangebins(idx);
    %     else
    %         hat_r_tars(k) = 0;
    %         hat_v_tars(k) = 0;
    %         doppler_freq = 0;
    %         Pxx = zeros(N_sym, 1);
    %         F = zeros(N_sym, 1);
    %         max_idx = 1;
    %     end
    %     subplot(num_of_targets, 1, k);
    %     h = plot(F, 10*log10(Pxx), 'k-', 'LineWidth', 1.5);
    %     hold on;
    %     h_peak = plot(doppler_freq, 10*log10(Pxx(max_idx)), 'ro', ...
    %                  'MarkerSize', 5, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    % 
    %     box on;
    %     set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
    %     set(gca, 'LineWidth', 1.5);
    % 
    %     xlabel('Doppler Frequency, $f_{D,i}$ (Hz)', 'Interpreter', 'latex', 'FontSize', 14);
    %     ylabel(['PSD $|Z^{\rm prbs}_{\mathrm{sen},', num2str(k), '}[k]|^2$'], ...
    %    'Interpreter', 'latex', 'FontSize', 14);
    % 
    %     % Simple title
    %     title(['Target $k_{', num2str(k), '}$ with $k_{', num2str(k), ...
    %    '} = \arg\max_{k \in \{0, \ldots, N_{\rm sym}-1\}} |Z^{\rm prbs}_{\mathrm{sen},', num2str(k), '}[k]|^2$'], ...
    %    'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'bold');
    % 
    %     % Get axis limits
    %     xlims = xlim;
    %     ylims = ylim;
    % 
    %     % Add velocity annotation text
    %     velocity_text = sprintf('$k_{%d}$: %d; $\\hat{\\nu}_{%d} = \\frac{-c k_{%d}}{2f_c N_{\\rm sym} T_{\\rm prbs}} = %.2f$ m/s', ...
    %                            k, max_idx, k, k, hat_v_tars(k));
    % 
    %     % Position velocity annotation
    %     text(xlims(1) + 0.05*(xlims(2)-xlims(1)), ...
    %          ylims(2) - 0.12*(ylims(2)-ylims(1)), ...
    %          velocity_text, ...
    %          'Interpreter', 'latex', 'FontSize', 10, 'FontName', 'Times New Roman', ...
    %          'BackgroundColor', [1 1 1 0.95], 'EdgeColor', [0.5 0.5 0.5], ...
    %          'LineWidth', 1, 'Margin', 4);
    % 
    %     % Add legend (hide connecting lines)
    %     legend([h, h_peak], {'Doppler Spectrum', 'Detected Peak'}, ...
    %            'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 11);
    % end
    % 
    % 
    % %% Range Estimation Plot
    % fig_range = figure('Position', [150, 150, 500, 500]);
    % set(fig_range, 'Color', 'w');
    % 
    % h1 = plot(rangebins, abs(z_time(:, 1)), 'k-', 'LineWidth', 1.5);
    % hold on;
    % 
    % h2 = plot(rangebins(best_peak(:, 1)), abs(z_time(best_peak(:, 1), 1)), ...
    %          'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    % 
    % box on;
    % set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
    % set(gca, 'LineWidth', 1.5);
    % 
    % xlabel('Range, $r$ (m)', 'Interpreter', 'latex', 'FontSize', 14);
    % ylabel('Amplitude, $|z_{\rm sen}^{\rm prbs}[k]|^2$', 'Interpreter', 'latex', 'FontSize', 14);
    % title('Range Profile with Detected Peaks', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'bold');
    % 
    % % Get axis limits
    % xlims = xlim;
    % ylims = ylim;
    % 
    % % Add formulation annotation
    % text(xlims(1) + 0.25*(xlims(2)-xlims(1)), ...
    %  ylims(2) - 0.3*(ylims(2)-ylims(1)), ...
    %  '$k = \arg\max_{k \in \{0, \ldots, N_{\rm prbs}-1\}} |z^{\rm prbs}_{\rm sen}[k]|^2$', ...
    %  'Interpreter', 'latex', 'FontSize', 12, ...
    %  'BackgroundColor', 'w', 'EdgeColor', 'k', ...
    %  'LineWidth', 1.2, 'Margin', 8);
    % 
    % 
    % 
    % % Add legend without data1, data2
    % legend([h1, h2], {'Received Signal', 'Detected Peaks'}, ...
    %        'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 11);
    % 
    % % Add target annotations with background
    % for k = 1:num_of_targets
    %     if best_peak(k) > 0
    %         peak_range = rangebins(best_peak(k));
    %         peak_amp = abs(z_time(best_peak(k), 1));
    % 
    %         % Annotation text
    %         annotation_text = sprintf('$k_{%d}$: %d; $\\hat{r}_{%d} = \\frac{c T_{\\rm prbs} k_%d}{2N_{\\rm prbs}} = %.2f$ m', ...
    %                                 k, best_peak(k), k, k, hat_r_tars(k));
    % 
    %         % Smart positioning based on peak location
    %         if peak_range < xlims(2) * 0.4
    %             % Left side peaks - annotate to the right
    %             text_x = peak_range + 0.15*(xlims(2)-xlims(1));
    %             h_align = 'left';
    %         else
    %             % Right side peaks - annotate to the left
    %             text_x = peak_range - 0.15*(xlims(2)-xlims(1));
    %             h_align = 'right';
    %         end
    % 
    %         % Vertical offset to avoid overlap
    %         text_y = peak_amp + 0.02*(ylims(2)-ylims(1)) + (k-1)*0.08*(ylims(2)-ylims(1));
    % 
    %         % Add text with background
    %         text(text_x, text_y, annotation_text, ...
    %              'FontSize', 10, 'FontName', 'Times New Roman', ...
    %              'Interpreter', 'latex', 'HorizontalAlignment', h_align, ...
    %              'VerticalAlignment', 'middle', 'BackgroundColor', [1 1 1 0.95], ...
    %              'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1, 'Margin', 4);
    % 
    %         % Add connecting line from text to peak
    %         line([text_x peak_range], [text_y peak_amp], ...
    %              'Color', [0.3 0.3 0.3], 'LineStyle', '--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    %     end
    % end
end