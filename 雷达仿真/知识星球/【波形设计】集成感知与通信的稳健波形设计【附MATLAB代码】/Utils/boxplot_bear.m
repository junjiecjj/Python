%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function boxplot_bear(X, group, widths, colors, linewidth, linetype)
    [~, ColLen] = size(X);
    Y = zeros(5, ColLen);
    for i = 1:ColLen
        % Define the desired percentiles (e.g., 10th and 90th percentiles)
        custom_percentiles = prctile(X(:, i), [5, 95]);
    
        % Create a modified version of data with custom percentiles
        Y(:, i) = [min(X(:, i)); custom_percentiles(1); median(X(:, i)); custom_percentiles(2); max(X(:, i))];
    end
    
    % Create the boxplot with modified data
    if strcmp(colors, '')
        h = boxplot(Y, group, 'Widths', widths);
    else
        h = boxplot(Y, group, 'Widths', widths, 'Colors', colors);
    end
    
        % h contains 7 MATLAB Line objects
        % Line    (Upper Whisker)
        % Line    (Lower Whisker)
        % Line    (Upper Adjacent Value)
        % Line    (Lower Adjacent Value)
        % Line    (Box)
        % Line    (Median)
        % Line    (Outliers)

    set(h, 'linewidth', linewidth);
    set(findobj(h, 'type', 'line'), 'LineStyle', linetype);
end

