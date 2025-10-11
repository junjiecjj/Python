function signal = generateDataGaussianWhite(num_cells, ...
                                            pos_target, ...
                                            echo_power_dB, ...
                                            noise_power_dB)
    % ======>INPUT:
    %   num_cells: The amount of cells.[DATATYPE: scalar]
    %   pos_target: Range of all targets.[DATATYPE: vector]
    %   echo_power_dB: Echo Wave power in dB.[DATATYPE: scalar]
    %   noise_power_dB: Noise power in dB.[DATATYPE: scalar]
    % ======>OUTPUT:
    %   signal: Mixed data of signal and noise.[DATATYPE: row vector]

    num_target = length(pos_target);
    echo_power = db2pow(echo_power_dB);
    noise_power = db2pow(noise_power_dB);

    noise_sample = randn(1, num_cells) + 1j * randn(1, num_cells);
    noise = abs(sqrt(noise_power/2) * noise_sample) .^ 2;

    if num_target == 0
        signal = noise;
    else
        signal_sample = zeros(1, num_cells);
        for ii = 1 : num_target
            signal_sample(pos_target(ii)) = echo_power(ii);
        end
        signal = signal_sample + noise;
    end

return