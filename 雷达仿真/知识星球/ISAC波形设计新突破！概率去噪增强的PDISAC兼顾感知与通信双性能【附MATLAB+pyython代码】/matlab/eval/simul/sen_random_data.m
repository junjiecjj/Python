% Signal Settings
fc = 24e9; % Carrier frequency (Hz)
Gtx = 50; % Tx antenna gain (dB)
Grx = 50; % Radar Rx antenna gain (dB)
Grx_ue = 30; % UE Rx antenna gain (dB)
NF = 2.9; % Noise figure (dB)
Tref = 290; % Reference temperature (K)
Rmax = 200; % Maximum range of interest
vrelmax = 60; % Maximum relative velocity
c = physconst('LightSpeed');
lambda = c / fc;


num_of_bits = 256;
num_of_chips = 255;

P_prbs = load("data/p_freq_255.mat").data;
p_prbs = ifft(P_prbs);


T_d = 5.1e-6;
T_chip = T_d / (2 * num_of_chips);
T_prbs = T_d / 2;
T_D = T_d * num_of_bits;
T_frame = T_D / 2;
fs = 1 / T_chip;
B = fs;

rdr = phased.RangeDopplerResponse(...
    'RangeMethod', 'FFT', ...
    'SampleRate', fs, ...
    'SweepSlope', -B/T_prbs, ...
    'DopplerOutput', 'Speed', ...
    'OperatingFrequency', fc, ...
    'PRFSource', 'Property', ...
    'PRF', 1/T_prbs, ...
'ReferenceRangeCentered', false);

num_refer = [10, 10]; % [Range, Doppler] refer cells
num_guard = [4, 4]; % [Range, Doppler] guard cells
pfa = 1e-4; % Probability of false alarm
cfar2D = phased.CFARDetector2D(...
    'GuardBandSize', num_guard, ...
    'TrainingBandSize', num_refer, ...
    'ProbabilityFalseAlarm', pfa);


% Setting hyperparameter for sensing
num_of_trials = 1000;
SNRs_Tx = [-20, -10, -5, 0, 5, 10, 20, 30, 40, 50];


NF_lin = 10^(NF / 10);
noise_power = physconst('Boltzmann') * Tref * NF_lin * B;


target_rcs = 4.7; 
source_dir = "data/";


fov_simulation(5, Rmax, vrelmax) 

% num_of_targets = 1;
% target_speed = -47.2851;
% target_distance = 91.5379;

for idx_SNR = 1:length(SNRs_Tx)
    SNR_tx = SNRs_Tx(idx_SNR);
    Pt = noise_power * 10^(SNR_tx/10);
    
    
    % Give trial for each Noise Figures
    for idx_trial = 1:num_of_trials
        
        y_PRBS_waveform_no_noise = 0;
        num_of_targets = randi(5);
        % num_of_targets = 5;

        for idx_tar = 1:num_of_targets

            [target_location, target_vel] = randomLV(Rmax, vrelmax);
            r_hat = target_location / vecnorm(target_location);      % LOS unit vector
            target_speed = vecnorm(target_vel) * sign(dot(target_vel, r_hat));
            target_distance = vecnorm(target_location);
            
            Gtx_lin = 10^(Gtx / 10); 
            Grx_lin = 10^(Grx / 10);
            alpha_tar =  sqrt((Pt * Gtx_lin * Grx_lin * lambda^2 * target_rcs) / ((4*pi)^3 * target_distance^4));
            y_PRBS_waveform_no_noise = y_PRBS_waveform_no_noise + generate_signal_model_parametric(target_distance, target_speed, alpha_tar, p_prbs, T_chip, T_d, fc, c, num_of_chips, num_of_bits);
            y_PRBS_waveform = add_complex_noise(y_PRBS_waveform_no_noise, noise_power);
        end

        
        Y_PRBS_waveform = fft(y_PRBS_waveform);
        Z_PRBS_waveform = Y_PRBS_waveform .* conj(P_prbs);
        z_PRBS_waveform = ifft(Z_PRBS_waveform);

        Y_PRBS_waveform_no_noise = fft(y_PRBS_waveform_no_noise);
        Z_PRBS_waveform_no_noise = Y_PRBS_waveform_no_noise .* conj(P_prbs);
        z_PRBS_waveform_no_noise = ifft(Z_PRBS_waveform_no_noise);
       

        % [hat_r_tars, hat_v_tars] = est_rv_seq_z(z_PRBS_waveform, T_prbs, num_of_targets, fs, lambda);
        % figure;
        % plotResponse(rdr, Z_PRBS_waveform_no_noise, 'Unit', 'db');
        % xlim([-vrelmax vrelmax]);
        % ylim([0 Rmax]);
        % hold on;
        % scatter(hat_v_tars, hat_r_tars, 100, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
        % hold off;


        if idx_trial < 3
            waveform_types = {'Z_PRBS_waveform', 'Y_PRBS_waveform'};
            waveform_data = {Z_PRBS_waveform, Y_PRBS_waveform};


            % Loop through each waveform type to generate and save plots
            for i = 1:length(waveform_types)
                % Create directory for waveform type and SNR
                output_dir = fullfile(source_dir, "images", waveform_types{i}, num2str(SNR_tx));
                if ~exist(output_dir, 'dir')
                    mkdir(output_dir);
                end

                % Create figure (no display)
                figure('visible', 'off');

                % Plot the waveform
                plotResponse(rdr, waveform_data{i}, 'Unit', 'db');
                xlim([-vrelmax vrelmax]);
                ylim([0 Rmax]);

                % Define filename and save as PNG
                filename = fullfile(output_dir, sprintf('%s_%d.png', waveform_types{i}, idx_trial));
                saveas(gcf, filename); % Save as PNG

                % Close the figure to free memory
                close(gcf);
            end
        end

        % ###############################
        % Save data in frequency domain #
        % ###############################
        waveform_types = {'Y_PRBS_waveform_no_noise', 'Y_PRBS_waveform'};
        waveform_data = {Y_PRBS_waveform_no_noise, Y_PRBS_waveform};


        % Loop through each waveform type to save data
        for i = 1:length(waveform_types)
            % Create directory for waveform type and SNR
            output_dir = fullfile(source_dir, "mats", "freq", waveform_types{i}, num2str(SNR_tx));
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end

            % Save waveform data as .mat file
            data = waveform_data{i};
            filename = fullfile(output_dir, sprintf('%d.mat', idx_trial));
            save(filename, 'data');
        end
    end

end


% Helper functions for Doppler and wavelength
function dop = speed2dop(speed, wavelength)
    dop = speed / wavelength;
end



function [l, v] = randomLV(Rmax, Vmax, z_fixed)

    if nargin < 3
        z_fixed = [];
    end

    Rmin = 10;   % minimum range

    % -------- Position vector --------
    dir_l = randn(3,1);
    dir_l = dir_l / norm(dir_l);                 % unit direction

    r = Rmin + (Rmax - Rmin) * rand();            % 20 ≤ r ≤ Rmax
    l = dir_l * r;

    % -------- Velocity vector --------
    dir_v = randn(3,1);
    dir_v = dir_v / norm(dir_v);                 % unit direction

    vmag = Vmax * rand();                        % 0 ≤ |v| ≤ Vmax
    v = dir_v * vmag;

    % -------- Optional fixed height --------
    if ~isempty(z_fixed)
        l(3) = z_fixed;   % fix target height
        v(3) = 0;        % ground motion
    end
end


function S_model = generate_signal_model_parametric(r, v, A, p_prbs, T_chip, T_sym, fc, c, N_fast, N_slow)
    % Generate parametric signal model following:
    % y[n,m] = A * p[n-n0] * exp(j*(2*pi*fd*m*T_sym - phi0))
    %
    % where:
    %   n0 = round(2*R / (c*T_chip)) - delay in chips
    %   fd = 2*v*fc/c - Doppler frequency (Hz)
    %   phi0 = 4*pi*fc*R/c - carrier phase shift (radians)

    % Step 1: Compute delay in chips (round-trip)
    tau = 2 * r / c; % Round-trip delay (seconds)
    n0 = round(tau / T_chip); % Integer delay in chips

    % Step 2: Shift PRBS code by delay
    % p[n - n0] means we shift p to the right by n0 samples
    % In MATLAB: circshift with negative means shift left, positive means shift right
    % We want p[n] -> p[n-n0], which is shifting the sequence to the right
    n0_wrapped = mod(n0, N_fast); % Wrap around for circular shift
    p_shifted = circshift(p_prbs, n0_wrapped); % Shift right by n0

    % Step 3: Compute Doppler frequency
    fd = -2 * v * fc / c; % Hz

    % Step 4: Compute carrier phase shift
    phi0 = 4 * pi * fc * r / c; % radians

    % Step 5: Generate phase modulation for slow-time
    % Phase varies as: 2*pi*fd*m*T_sym for m = 0, 1, 2, ..., N_slow-1
    m_indices = 0:(N_slow-1); % Slow-time indices
    phase_doppler = 2 * pi * fd * m_indices * T_sym; % 1 x N_slow

    % Step 6: Combine all components
    % y[n,m] = A * p_shifted[n] * exp(j*(phase_doppler[m] - phi0))

    % Create phase matrix: subtract phi0 from each Doppler phase
    phase_total = phase_doppler - phi0; % 1 x N_slow

    % Generate complex exponential
    exp_phase = exp(1j * phase_total); % 1 x N_slow

    % Combine: p_shifted (N_fast x 1) with exp_phase (1 x N_slow)
    % Broadcasting: (N_fast x 1) * (1 x N_slow) -> (N_fast x N_slow)
    S_model = A * p_shifted * exp_phase; % N_fast x N_slow
end

function x_noisy = add_complex_noise(x, noise_power)
    % add_complex_noise - Adds complex Gaussian noise CN(0, sigma^2) to signal
    %
    % Inputs:
    %   x           : Input signal (N x M)
    %   noise_power : Noise power sigma^2 (scalar)
    %
    % Output:
    %   x_noisy : Noisy signal x + n (N x M)
    
    [N, M] = size(x);
    
    % Generate complex Gaussian noise CN(0, sigma^2)
    % For CN(0, sigma^2): real and imaginary parts are each N(0, sigma^2/2)
    noise_real = sqrt(noise_power/2) * randn(N, M);
    noise_imag = sqrt(noise_power/2) * randn(N, M);
    
    n = noise_real + 1i * noise_imag;
    
    % Add noise to signal
    x_noisy = x + n;
    
end

function fov = fov_simulation(num_of_targets, Rmax, vrelmax)
    target_locations = zeros(3, num_of_targets);
    target_velocities = zeros(3, num_of_targets);
    
    for idx_tar = 1:num_of_targets
        [target_locations(:, idx_tar), target_velocities(:, idx_tar)] = ...
            randomLV(Rmax, vrelmax);
    end
    
    % Plot the radar and targets
    opts.normalizeArrows = true;  % All arrows same length
    opts.arrowFrac = 0.1;         % Larger arrows
    opts.showFOV = true;         % Hide FOV sphere
    plot_radar_vs_targets(target_locations, target_velocities, Rmax, opts);
end