function helperPlotSpec(axis_angle, spec, name, varargin)
    % Parse optional line style argument
    if ~isempty(varargin)
        lineStyle = varargin{1};
    else
        lineStyle = '-'; % Default line style
    end

    % Plot spectra in dB normalized to 0 dB at the peak
    y_db = 20 * log10(spec) - max(20 * log10(spec));
    plot(axis_angle, y_db, lineStyle, 'DisplayName', name)

    xlabel('Broadside Angle (degrees)');
    ylabel('Power (dB)');
    title('DOA Spatial Spectra')
    legend;
    grid on;
end
