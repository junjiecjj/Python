function helperPlotSpec2d(x, y, spec, name)

    % Plot spectra in dB normalized to 0 dB at the peak
    Z = 20 * log10(spec);
    Z = Z -max(Z, [], 'all');
    [X, Y] = meshgrid(x, y);
    mesh(X, Y, Z, 'DisplayName', name)

    xlabel('Azimuth (Degrees)');
    ylabel('Range (m)');
    zlabel('Power (dB)');
    title('DOA Spatial Spectra');
    legend;
end
