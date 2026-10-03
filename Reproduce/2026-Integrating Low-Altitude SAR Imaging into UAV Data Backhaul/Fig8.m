%% Fig.8: public SAR image -> synthetic OFDM echoes -> RD reconstruction
% Dataset: CAESAR-Radi/SAR-Ship-Dataset, cited as [46] in the paper.
% This is a same-dataset reproduction; the exact Fig.8(a) filename is unknown.
% Supplements: image preprocessing, complex phases, coordinates and noise scale.
% Forward model uses a common reference range in the migration/azimuth terms.
% It retains each pixel's own range phase. This is a reference-range approximation
% to Eq.(9), not an exact sum using a separate Ka for every scatterer.
% No image blur is used to create reconstructed results.
% MATLAB R2016b or later; Image Processing Toolbox is not required.

clear;
clc;
close all;
rng(42, 'twister');

%% 1. Read the locally downloaded SAR image
% JPG is the scene input. YOLO TXT labels are not required for SAR imaging.
dataset_root = '/home/jack/图片/ship_dataset_v0';
scene_file = fullfile(dataset_root, 'Gao_ship_hh_02016082544040402.jpg');
% scene_file = fullfile(dataset_root, 'CD.png');
auto_download = false;
scene_index = 1; % Used only when scene_file is empty.

scene_file = resolve_image(scene_file, dataset_root, auto_download, scene_index);
fprintf('Scene image: %s\n', scene_file);
raw = imread(scene_file);
if ndims(raw) == 3
    raw = 0.2989 * double(raw(:, :, 1)) + ...
        0.5870 * double(raw(:, :, 2)) + 0.1140 * double(raw(:, :, 3));
else
    raw = double(raw);
end

%% 2. System and explicitly supplemented scene settings
P.c = 3e8;
P.fc = 3.5e9;
P.Deltaf = 30e3;
P.Tsym = 1 / P.Deltaf + 8.33e-6;
P.N = round(100e6 / P.Deltaf);
P.rhor = P.c / (2 * P.N * P.Deltaf);
P.v = 50;
P.Hp = 1000;
P.Ta = 2;
P.SNRin_dB = 5;
P.SNRin = 10^(P.SNRin_dB / 10);
P.scene_rows = 256;
P.scene_columns = 256;
P.y_center = 100;
P.azimuth_padding_factor = 2;
P.block_size = 32;
P.phase_mode = 'random'; % 'random' or 'zero'; original paper does not specify.
P.noise_mode = 'masked_physical'; % Optional diagnostic: 'paper_white_approx'.
P.display_range_dB = 45;

% Physical slant range must be at least Hp. Original Fig.8 coordinate mapping
% is not specified; use a valid supplemental grid instead of copying its labels.
P.k_scene = ceil((P.Hp + 5) / P.rhor) + (0:P.scene_rows - 1).';
P.Rscene = P.k_scene * P.rhor;
P.Rref = mean(P.Rscene);
P.Ka = 2 * P.v^2 / ((P.c / P.fc) * P.Rref);

I = resize_linear(raw, P.scene_rows, P.scene_columns);
I = max(I, 0);
if max(I(:)) <= 0
    error('The selected image is empty or completely black.');
end
% Supplement: interpret normalized displayed gray values as relative power.
% These are not calibrated linear satellite backscatter measurements.
I = I / max(I(:));
if strcmp(P.phase_mode, 'random')
    phase = 2 * pi * rand(size(I));
elseif strcmp(P.phase_mode, 'zero')
    phase = zeros(size(I));
else
    error('Unknown phase mode.');
end
alpha = sqrt(I) .* exp(1i * phase);
alpha = alpha / norm(alpha, 'fro');
P.sigma2 = sum(abs(alpha(:)).^2) / P.SNRin;

%% 3. Resource configurations from Section V-C
pilot_n = (1667:4:1954).'; % Zero-based subcarrier indices.
active_n = {pilot_n, pilot_n, (0:P.N - 1).'};
symbol_steps = [280, 28, 10];
titles = {'(a) Reference scene', '(b) Pilot-only: 20 slots', ...
    '(c) Pilot-only: 2 slots', '(d) Data-aided imaging'};

fig = figure('Color', 'w', 'Position', [50, 160, 1500, 430]);
ax = subplot(1, 4, 1, 'Parent', fig);
y_reference = linspace(0, 200, P.scene_columns);
show_image(ax, y_reference, P.Rscene, alpha, P.display_range_dB, titles{1});
drawnow;

reconstructed_images = cell(1, 3);
output_azimuth_axes = cell(1, 3);
for scheme = 1:3
    fprintf('\n%s\n', titles{scheme + 1});
    [image_out, y_out] = reconstruct_scene(P, alpha, active_n{scheme}, symbol_steps(scheme), scheme == 3);
    reconstructed_images{scheme} = image_out;
    output_azimuth_axes{scheme} = y_out;
    ax = subplot(1, 4, scheme + 1, 'Parent', fig);
    show_image(ax, y_out, P.Rscene, image_out, P.display_range_dB, titles{scheme + 1});
    drawnow;
end

fprintf('\nAll panels use the same scattering scene and per-panel peak normalization.\n');
fprintf('Range extent: %.3f to %.3f m; reference range: %.3f m\n', ...
    P.Rscene(1), P.Rscene(end), P.Rref);
fprintf('Phase mode: %s; noise mode: %s\n', P.phase_mode, P.noise_mode);
fprintf('This is not an exact reconstruction of the original Fig.8 sample.\n');

%% Local functions
function filename = resolve_image(filename, root, auto_download, scene_index)
    if ~isempty(filename)
        if ~isfile(filename)
            error('The specified scene_file does not exist.');
        end
        return;
    end
    files = find_images(root);
    if isempty(files) && auto_download
        if ~isfolder(root)
            mkdir(root);
        end
        archive = fullfile(root, 'ship_dataset_v0.zip');
        url = ['https://github.com/CAESAR-Radi/SAR-Ship-Dataset/' ...
            'raw/refs/heads/2021-04-update/ship_dataset_v0.zip'];
        try
            if ~isfile(archive)
                fprintf('Downloading the public archive (approximately 407 MB)...\n');
                websave(archive, url, weboptions('Timeout', 120));
            end
            fprintf('Extracting the dataset...\n');
            unzip(archive, root);
            files = find_images(root);
        catch exception
            fprintf('Automatic download/extraction failed: %s\n', exception.message);
            fprintf('Select a local SAR image in the following dialog.\n');
        end
    end
    if ~isempty(files)
        if scene_index < 1 || scene_index > numel(files) || scene_index ~= floor(scene_index)
            error('scene_index must be between 1 and %d.', numel(files));
        end
        filename = fullfile(files(scene_index).folder, files(scene_index).name);
    else
        [name, folder] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', ...
            'SAR image'}, 'Select one SAR scene image');
        if isequal(name, 0)
            error('No image selected. Set scene_file or download the dataset first.');
        end
        filename = fullfile(folder, name);
    end
end

function files = find_images(root)
    files = struct('name', {}, 'folder', {});
    if ~isfolder(root)
        return;
    end
    entries = dir(fullfile(root, '**', '*'));
    for ii = 1:numel(entries)
        [~, ~, extension] = fileparts(entries(ii).name);
        if ~entries(ii).isdir && any(strcmpi(extension, ...
                {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}))
            files(end + 1).name = entries(ii).name;
            files(end).folder = entries(ii).folder;
        end
    end
    if ~isempty(files)
        paths = arrayfun(@(f) fullfile(f.folder, f.name), files, 'UniformOutput', false);
        [~, order] = sort(paths);
        files = files(order);
    end
end

function resized = resize_linear(original, rows, columns)
    [x, y] = meshgrid(linspace(1, size(original, 2), columns), ...
        linspace(1, size(original, 1), rows));
    resized = interp2(original, x, y, 'linear');
end

function [image_out, y_out] = reconstruct_scene(P, alpha, active_n, symbol_step, is_data)
    N = P.N;
    Tslow = symbol_step * P.Tsym;
    M = floor((P.Ta - eps(P.Ta)) / Tslow) + 1;
    Lfft = P.azimuth_padding_factor * M;
    mq = floor(Lfft / 2);
    observed_columns = mq + (0:M - 1) - floor(M / 2) + 1;
    y_full = P.y_center + P.v * ((0:Lfft - 1) - mq) * Tslow;
    y_desired = linspace(0, 200, size(alpha, 2));
    scene_columns = round((y_desired - y_full(1)) / (P.v * Tslow)) + 1;
    scene_columns = min(max(scene_columns, 1), Lfft);
    if numel(unique(scene_columns)) ~= numel(scene_columns)
        error('Scene azimuth grid is too dense for this sampling scheme.');
    end
    y_out = y_full(scene_columns);
    fprintf('Maximum scene-grid quantization error: %.4f m\n', max(abs(y_out - y_desired)));
    fprintf('Observed symbols = %d, active subcarriers = %d, PRF = %.3f Hz\n', ...
        M, numel(active_n), 1 / Tslow);

    % Efficient forward model: linear chirp convolution, not circular convolution.
    % H(n,m) = sum_r,j alpha(r,j) exp(-j4pi*n*df*Rr/c)
    %          * exp(-j2pi*(fc+n*df)*(ym-yj)^2/(c*Rref)).
    % This is Eq.(9) with Rref in DeltaR and each Rr in the range term.
    lags = -(Lfft - 1):(Lfft - 1);
    dy2 = (P.v * Tslow * lags).^2;
    convolution_length = 2^nextpow2(3 * Lfft - 2);
    extraction = observed_columns + Lfft - 1;
    Ytf = complex(zeros(N, Lfft));

    for first = 1:P.block_size:numel(active_n)
        n = active_n(first:min(first + P.block_size - 1, numel(active_n)));
        n = n(:);
        Kb = numel(n);
        range_projection = exp(-1i * 4 * pi / P.c * ...
            (n * P.Deltaf) * P.Rscene.') * alpha;
        source = complex(zeros(Kb, Lfft));
        source(:, scene_columns) = range_projection;
        kernel = exp(-1i * 2 * pi / (P.c * P.Rref) * ...
            (P.fc + n * P.Deltaf) * dy2);
        convolution = ifft(fft(source, convolution_length, 2) .* ...
            fft(kernel, convolution_length, 2), [], 2);
        H = convolution(:, extraction);

        if is_data
            if exist('qammod', 'file') == 2
                S = qammod(randi([0, 255], Kb, M), 256, 'UnitAveragePower', true);
            else
                S = (2 * randi([0, 15], Kb, M) - 15 + ...
                    1i * (2 * randi([0, 15], Kb, M) - 15)) / sqrt(170);
            end
        else
            S = exp(1i * pi / 2 * randi([0, 3], Kb, M));
        end
        Z = sqrt(P.sigma2 / 2) * (randn(Kb, M) + 1i * randn(Kb, M));
        Y = H .* S + Z;
        Ytf(n + 1, observed_columns) = Y .* conj(S); % MF Eq.(20).
    end
    clear source kernel convolution H S Z Y;

    if ~is_data && strcmp(P.noise_mode, 'paper_white_approx')
        inactive_n = setdiff((0:N - 1).', active_n);
        for first = 1:P.block_size:numel(inactive_n)
            n = inactive_n(first:min(first + P.block_size - 1, numel(inactive_n)));
            Ytf(n + 1, observed_columns) = sqrt(P.sigma2 / 2) * ...
                (randn(numel(n), M) + 1i * randn(numel(n), M));
        end
    elseif ~strcmp(P.noise_mode, 'masked_physical') && ...
            ~strcmp(P.noise_mode, 'paper_white_approx')
        error('Unknown noise mode.');
    end

    %% Original RD processing with the same reference-range approximation
    Yrc = sqrt(N) * ifft(Ytf, [], 1);
    clear Ytf;
    Yrd = fftshift(fft(Yrc, [], 2), 2) / sqrt(Lfft);
    clear Yrc;
    p = -floor(Lfft / 2):ceil(Lfft / 2) - 1;
    fD = p / (Lfft * Tslow);
    Delta_k = P.v^2 * fD.^2 / (2 * P.Rref * P.Ka^2 * P.rhor);
    Haz = exp(-1i * pi * fD.^2 / P.Ka);
    n_all = (0:N - 1).';
    corrected = complex(zeros(numel(P.k_scene), Lfft));

    for first = 1:P.block_size:Lfft
        columns = first:min(first + P.block_size - 1, Lfft);
        spectrum = fft(Yrd(:, columns), [], 1) / sqrt(N);
        shifted = sqrt(N) * ifft(spectrum .* ...
            exp(1i * 2 * pi / N * n_all * Delta_k(columns)), [], 1);
        corrected(:, columns) = shifted(P.k_scene + 1, :) .* Haz(columns);
    end
    clear Yrd;
    focused = sqrt(Lfft) * ifft(ifftshift(corrected, 2), [], 2);
    % Extract native scene locations, using the same selection for all panels.
    image_out = focused(:, scene_columns);
end

function show_image(ax, azimuth, range, data, dynamic_range_dB, panel_title)
    magnitude = abs(data);
    reference = max(magnitude(:));
    image_dB = 20 * log10(max(magnitude / max(reference, eps), eps));
    imagesc(ax, azimuth, range, image_dB);
    set(ax, 'YDir', 'normal', 'FontName', 'Times New Roman', 'FontSize', 10);
    colormap(ax, hot(256));
    caxis(ax, [-dynamic_range_dB, 0]);
    xlabel(ax, 'Azimuth (m)');
    ylabel(ax, 'Slant range (m)');
    title(ax, panel_title);
    xlim(ax, [0, 200]);
    box(ax, 'on');
end