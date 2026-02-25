function normalized_data = normalize_waveforms(waveform_data, a, b)
    % Initialize output cell array with the same size as input
    normalized_data = cell(size(waveform_data));
    
    % Process each matrix in the cell array
    for i = 1:length(waveform_data)
        % Get the current complex matrix
        Z = waveform_data{i};
        
        % Extract real and imaginary parts
        real_part = real(Z);
        imag_part = imag(Z);
        
        % Compute min and max for real parts
        min_real = min(real_part(:));
        max_real = max(real_part(:));
        
        % Compute min and max for imaginary parts
        min_imag = min(imag_part(:));
        max_imag = max(imag_part(:));
        
        % Normalize real parts to [a, b]
        if max_real ~= min_real
            real_normalized = a + (real_part - min_real) * (b - a) / (max_real - min_real);
        else
            real_normalized = real_part; % Or set to (a + b)/2 if desired
        end
        
        % Normalize imaginary parts to [a, b]
        if max_imag ~= min_imag
            imag_normalized = a + (imag_part - min_imag) * (b - a) / (max_imag - min_imag);
        else
            imag_normalized = imag_part; % Or set to (a + b)/2 if desired
        end
        
        % Recombine into a complex matrix
        Z_normalized = real_normalized + 1i * imag_normalized;
        
        % Store in the output cell array
        normalized_data{i} = Z_normalized;
    end
end