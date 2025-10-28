%% ============================================================
%  FMCW Radar: Range / Velocity / Angle All-in-One Demo
%  - Angle–Range uses per-range strongest Doppler slice
%  - Legends + Truth Annotations on RD, RA, 3D
%  Single-file, one-click runnable
%  Tested: R2020b+ (no special toolboxes required)
% =============================================================

clear; clc; close all;

%% ----------------------- User Parameters -----------------------
% Radar & waveform
c        = 3e8;             % speed of light
fc       = 77e9;            % carrier frequency (Hz)
lambda   = c/fc;
BW       = 1e9;             % sweep bandwidth (Hz)
Tc       = 40e-6;           % chirp duration (s)
S        = BW/Tc;           % chirp slope (Hz/s)
fs       = 20e6;            % ADC sampling rate (Hz)
Ns       = round(fs*Tc);    % samples per chirp
Nchirp   = 128;             % number of chirps (slow-time)
Tx       = 1;               % 1-Tx (TDM off)
Rx       = 8;               % number of Rx (ULA)
d        = lambda/2;        % element spacing

% Scene (multi-target): [R(m), v(m/s), azimuth(deg), RCS/amp]
targets = [...
    40,   -10,   -10,  1.0;  % T1
    55,     8,    20,  0.8;  % T2
    85,     0,     5,  0.6   % T3
];

SNR_dB  = 20;               % SNR of dechirped baseband

% Processing params
Nfft_r  = 2^nextpow2(Ns*2);         % range FFT size (zero-pad)
Nfft_d  = 2^nextpow2(Nchirp*2);     % doppler FFT size
Nfft_a  = 256;                      % angle FFT size (beamforming)
use_MUSIC = true;                   % also compute MUSIC AoA (optional)
MUSIC_K   = size(targets,1);        % number of sources assumed for MUSIC
guard_r   = 2;  train_r = 8;        % 2D-CFAR params (range)
guard_d   = 2;  train_d = 8;        % 2D-CFAR params (doppler)
Pfa       = 1e-3;                   % CFAR false alarm

% Derived limits
R_max   = fs*c/(2*S);               % unambiguous range
V_max   = lambda/(4*Tc);            % +/- V_max unambiguous
fprintf('Unambiguous Range ~ %.1f m, Unambiguous Velocity ~ +/- %.1f m/s\n',...
        R_max, V_max);

%% ----------------------- Time & Grids -------------------------
t_fast = (0:Ns-1)/fs;               % fast-time within a chirp
n_slow = (0:Nchirp-1).';            % chirp index (slow-time)
f_fast = (0:Nfft_r-1)*(fs/Nfft_r);  % beat freq axis (0..fs)
R_axis = c * f_fast / (2*S);        % beat freq -> range

% Slow-time (Doppler) axis: PRF = 1/Tc, centered
f_doppler = (-Nfft_d/2:Nfft_d/2-1)*(1/(Tc*Nfft_d));
V_axis    = (lambda/2) * f_doppler; 

% Angle axis for Rx-ULA (broadside 0 deg; left negative)
mu_axis   = linspace(-1,1,Nfft_a);
ang_axis  = asind(mu_axis);

%% ----------------------- Signal Simulation --------------------
% Data cube: Rx x Nchirp x Ns (dechirped baseband)
X = zeros(Rx, Nchirp, Ns);

for k = 1:size(targets,1)
    R0   = targets(k,1);
    vk   = targets(k,2);
    thet = targets(k,3) * pi/180;
    amp  = targets(k,4);

    Rn   = R0 + vk * n_slow * Tc;           % Nchirp x 1
    tau  = 2 * Rn / c;                      % Nchirp x 1
    fd   = 2*vk/lambda;                     % Doppler (Hz)

    m_idx = (0:Rx-1).';
    phi_arr_m = 2*pi * (m_idx * d * sin(thet) / lambda); % Rx x 1

    S_tau = S * tau;                         % Nchirp x 1
    [S_tau_mat, t_fast_mat] = ndgrid(S_tau, t_fast);
    [n_slow_mat, ~]         = ndgrid(n_slow, t_fast);
    phase_nt = 2*pi*((S_tau_mat + fd).*t_fast_mat + fd .* n_slow_mat * Tc); % Nchirp x Ns

    for m = 1:Rx
        X(m,:,:) = squeeze(X(m,:,:)) + amp * exp(1j*(phase_nt + phi_arr_m(m)));
    end
end

% Add AWGN
sig_pow  = mean(abs(X(:)).^2);
noise_pw = sig_pow / (10^(SNR_dB/10));
noise    = sqrt(noise_pw/2) * (randn(size(X)) + 1j*randn(size(X)));
X = X + noise;

%% ----------------------- Range FFT ----------------------------
win_r = hann(Ns).';
X_r = zeros(Rx, Nchirp, Nfft_r);
for m = 1:Rx
    for n = 1:Nchirp
        x = squeeze(X(m,n,:)).' .* win_r;
        X_r(m,n,:) = fft(x, Nfft_r);
    end
end

valid_r = R_axis <= R_max;
R_hat   = R_axis(valid_r);
Nr      = numel(R_hat);

%% ----------------------- Doppler FFT (fixed) ------------------
win_d = hann(Nchirp);
Nd    = Nfft_d;
X_rd  = zeros(Rx, Nd, Nr);          % Rx x Nd x Nr

for m = 1:Rx
    Xr = squeeze(X_r(m,:,:));            % Nchirp x Nfft_r
    Xr = Xr(:, valid_r);                  % Nchirp x Nr
    Xr = Xr .* repmat(win_d, 1, size(Xr,2));
    Xrd = fftshift(fft(Xr, Nd, 1), 1);    % Nd x Nr
    X_rd(m,:,:) = reshape(Xrd, [1, size(Xrd,1), size(Xrd,2)]);
end

% Non-coherent sum over Rx for detection
RD = squeeze(sum(abs(X_rd).^2, 1));       % Nd x Nr

%% ----------------------- 2D-CFAR ------------------------------
mag = RD;
Nref = (2*train_r*2*train_d);
alpha = ca_cfar_alpha(Nref, Pfa);

det_map = false(Nd, Nr);
for id = 1+train_d+guard_d : Nd - (train_d+guard_d)
    d_idx = [id-(train_d+guard_d):id-guard_d-1, id+guard_d+1:id+train_d+guard_d];
    for ir = 1+train_r+guard_r : Nr - (train_r+guard_r)
        r_idx = [ir-(train_r+guard_r):ir-guard_r-1, ir+guard_r+1:ir+train_r+guard_r];
        noise_est = mean(mag(d_idx, r_idx), 'all');
        if mag(id, ir) > alpha * (noise_est + eps)
            det_map(id, ir) = true;
        end
    end
end
peak_idx = nms_peaks(mag, det_map, [3 3]);   % [id, ir]

%% ----------------------- Angle Estimation ---------------------
est_list = [];  % [R, V, AoA_FFT, Mag, AoA_MUSIC]
for p = 1:size(peak_idx,1)
    id = peak_idx(p,1);   % doppler bin
    ir = peak_idx(p,2);   % range bin

    fd_hat = f_doppler(id);
    V_est  = (lambda/2) * fd_hat;
    R_est  = R_hat(ir);

    x_m = squeeze(X_rd(:, id, ir));           % Rx x 1 snapshot

    ang_spectrum = abs(fftshift(fft(x_m, Nfft_a))).^2;
    [~, ia] = max(ang_spectrum);
    ang_fft = ang_axis(ia);

    ang_music = NaN;
    if use_MUSIC && Rx >= (MUSIC_K+1)
        ang_music = music_aoa(x_m, d, lambda, Nfft_a, MUSIC_K);
    end

    est_list = [est_list; R_est, V_est, ang_fft, norm(x_m), ang_music]; %#ok<AGROW>
end

%% ----------------------- Truth (for annotations) ---------------
R_true  = targets(:,1);
v_true  = targets(:,2);
th_true = targets(:,3);
fd_true = 2*v_true/lambda;           % Hz

%% ----------------------- Visualization ------------------------
% 1) Range profile
RP = squeeze(sum(sum(abs(X_r(:,:,valid_r)).^2, 1), 2)); % 1 x Nr
figure; plot(R_hat, 10*log10(RP/max(RP)+1e-12), 'LineWidth',1.3);
grid on; xlabel('Range (m)'); ylabel('Normalized Power (dB)');
title('Range Profile');

% 2) Range-Doppler map + detections + TRUTH + LEGEND
figure; 
imagesc(R_hat, f_doppler, 10*log10(mag./max(mag(:))+1e-12));
axis xy; colorbar; xlabel('Range (m)'); ylabel('Doppler (Hz)');
title('Range-Doppler Map (sum over Rx)');
hold on;
hDet = []; 
if ~isempty(peak_idx)
    hDet = scatter(R_hat(peak_idx(:,2)), f_doppler(peak_idx(:,1)), 40, 'w', 'filled', ...
        'DisplayName','Detections');
end
hTruth = plot(R_true, fd_true, 'p', 'MarkerSize', 12, 'MarkerEdgeColor', 'k', ...
     'MarkerFaceColor', [1 1 0], 'LineWidth', 1.5, 'DisplayName','Truth');
for i = 1:numel(R_true)
    text(R_true(i)+0.5, fd_true(i), sprintf('T%d', i), ...
        'Color', 'k', 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end
% Legend
if isempty(hDet)
    legend(hTruth, 'Location','best'); 
else
    legend([hDet, hTruth], {'Detections','Truth'}, 'Location','best');
end
hold off;

% 3) Angle spectrum for top-1 detection (FFT beamforming)
if ~isempty(peak_idx)
    id = peak_idx(1,1); ir = peak_idx(1,2);
    x_m = squeeze(X_rd(:, id, ir));
    ang_spec = abs(fftshift(fft(x_m, Nfft_a))).^2;
    ang_spec = ang_spec / max(ang_spec + 1e-12);
    figure; plot(ang_axis, 10*log10(ang_spec+1e-12), 'LineWidth',1.3); grid on;
    xlabel('Azimuth (deg)'); ylabel('dB'); 
    title('Angle Spectrum (Top-1, FFT Beamforming)');
end

% 4) Angle–Range 2D Map using "per-range strongest Doppler slice" + TRUTH + LEGEND
[~, id_max_per_r] = max(RD, [], 1);   % 1 x Nr (each range bin's strongest Doppler bin index)
RA = zeros(Nfft_a, Nr);
for ir = 1:Nr
    x_m = squeeze(X_rd(:, id_max_per_r(ir), ir));    % Rx x 1
    ang_spec = abs(fftshift(fft(x_m, Nfft_a))).^2;   % Nfft_a x 1
    RA(:, ir) = ang_spec;
end
RA = RA / (max(RA(:)) + 1e-12);

figure; 
imagesc(R_hat, ang_axis, 10*log10(RA + 1e-12));
axis xy; colorbar; xlabel('Range (m)'); ylabel('Azimuth (deg)');
title('Angle–Range Map (per-range strongest Doppler slice)');
hold on;
% truth overlay: only plot if target's fd is close to the chosen slice at its nearest range bin
df = abs(f_doppler(2) - f_doppler(1));    % Doppler frequency resolution
tol = 1.5*df;                             % tolerance: ~1.5 bins
truth_mask = false(size(R_true));
for i = 1:numel(R_true)
    [~, ir0] = min(abs(R_hat - R_true(i)));               % nearest range-bin to target range
    fd_slice = f_doppler(id_max_per_r(ir0));              % doppler slice used for this range
    truth_mask(i) = abs(fd_true(i) - fd_slice) <= tol;
end
hTruthRA = plot(R_true(truth_mask), th_true(truth_mask), 'p', 'MarkerSize', 11, ...
     'MarkerEdgeColor','k','MarkerFaceColor',[1 1 0],'LineWidth',1.3, 'DisplayName','Truth near slice');
for i = 1:numel(R_true)
    if truth_mask(i)
        text(R_true(i)+0.5, th_true(i), sprintf('T%d',i), ...
            'Color','k','FontWeight','bold','VerticalAlignment','middle');
    end
end
legend(hTruthRA, 'Location','best');
hold off;

% 5) 3D Scatter of Detections in (Range, Velocity, Angle) + TRUTH + LEGEND
if ~isempty(est_list)
    ang_use = est_list(:,5);                 % prefer MUSIC if available
    nan_idx = isnan(ang_use);
    ang_use(nan_idx) = est_list(nan_idx,3); % fallback to FFT AoA

    mag_lin = est_list(:,4);
    mag_db  = 20*log10(mag_lin / (max(mag_lin)+1e-12));
    sz = 30 + 70*(mag_lin / (max(mag_lin)+1e-12));  % marker size by magnitude

    figure;
    hDet3 = scatter3(est_list(:,1), est_list(:,2), ang_use, sz, mag_db, 'filled', ...
        'DisplayName','Detections');
    grid on; xlabel('Range (m)'); ylabel('Velocity (m/s)'); zlabel('Azimuth (deg)');
    title('3D Detection Cloud: Range–Velocity–Angle');
    cb = colorbar; cb.Label.String = 'Magnitude (dB, relative)';
    view(45,25);
    hold on;
    hTruth3 = plot3(R_true, v_true, th_true, 'p', 'MarkerSize', 12, ...
          'MarkerEdgeColor','k','MarkerFaceColor',[1 1 0], 'LineWidth',1.5, ...
          'DisplayName','Truth');
    for i = 1:numel(R_true)
        text(R_true(i), v_true(i), th_true(i)+1.0, sprintf('T%d',i), ...
            'Color','k','FontWeight','bold');
    end
    legend([hDet3 hTruth3], {'Detections','Truth'}, 'Location','best');
    hold off;
end

%% ----------------------- Print Detections ---------------------
if isempty(est_list)
    fprintf('No detections. Try lowering threshold (increase Pfa) or raising SNR.\n');
else
    % Sort by magnitude (desc)
    [~, ord] = sort(est_list(:,4), 'descend');
    est_sorted = est_list(ord,:);

    fprintf('\nDetections (after 2D-CFAR + AoA):\n');
    if use_MUSIC
        fprintf('   %9s  %10s  %10s  %10s  %12s  %10s\n','Range(m)','Vel(m/s)','AoA_FFT','Mag','AoA_MUSIC','Mag(dB)');
        for i=1:size(est_sorted,1)
            mag_db = 20*log10(est_sorted(i,4)/(max(est_sorted(:,4))+1e-12));
            fprintf('   %9.2f  %10.2f  %10.1f  %10.2f  %12.1f  %10.1f\n', ...
                est_sorted(i,1), est_sorted(i,2), est_sorted(i,3), est_sorted(i,4), est_sorted(i,5), mag_db);
        end
    else
        fprintf('   %9s  %10s  %10s  %10s  %10s\n','Range(m)','Vel(m/s)','AoA_FFT','Mag','Mag(dB)');
        for i=1:size(est_sorted,1)
            mag_db = 20*log10(est_sorted(i,4)/(max(est_sorted(:,4))+1e-12));
            fprintf('   %9.2f  %10.2f  %10.1f  %10.2f  %10.1f\n', ...
                est_sorted(i,1), est_sorted(i,2), est_sorted(i,3), est_sorted(i,4), mag_db);
        end
    end
end

%% ----------------------- Helper Functions ---------------------
function alpha = ca_cfar_alpha(Nref, Pfa)
% Return CA-CFAR scaling alpha for given reference cells and Pfa
% Using approximate formula: Pfa = (1+alpha/Nref)^(-Nref)
% => alpha = Nref * (Pfa^(-1/Nref) - 1)
alpha = Nref * (Pfa^(-1/Nref) - 1);
end

function peaks = nms_peaks(mag, det_map, nms_sz)
% Simple Non-Maximum Suppression on det_map using local neighborhood nms_sz
% mag: Nd x Nr, det_map: logical same size, nms_sz = [h w]
[H, W] = size(mag);
hh = floor(nms_sz(1)/2);
ww = floor(nms_sz(2)/2);
idx = find(det_map);
keep = true(size(idx));
for k = 1:length(idx)
    if ~keep(k), continue; end
    [r, c] = ind2sub([H,W], idx(k));
    r1 = max(1, r-hh); r2 = min(H, r+hh);
    c1 = max(1, c-ww); c2 = min(W, c+ww);
    patch = mag(r1:r2, c1:c2);
    if mag(r,c) < max(patch(:)) - eps
        keep(k) = false;
        continue;
    end
    [rr, cc] = find(det_map(r1:r2, c1:c2));
    rr = rr + (r1-1); cc = cc + (c1-1);
    nbr_lin = sub2ind([H,W], rr, cc);
    for t = 1:numel(nbr_lin)
        if nbr_lin(t) == idx(k), continue; end
        j = find(idx == nbr_lin(t), 1);
        if ~isempty(j), keep(j) = false; end
    end
end
sel = idx(keep);
[rr2, cc2] = ind2sub([H,W], sel);
peaks = [rr2, cc2];
end

function ang_hat = music_aoa(x_m, d, lambda, Nfft_a, K)
% x_m: Rx x 1 snapshot at a single range-doppler bin
% Returns peak of MUSIC spectrum (deg)
M = numel(x_m);
Rxx = (x_m * x_m')/M;
Rxx = Rxx + (1e-3*trace(Rxx)/M)*eye(M);
[V, D] = eig((Rxx+Rxx')/2);
[~, idx] = sort(real(diag(D)), 'ascend');
K = min(K, M-1);
En = V(:, idx(1:M-K));
mu_axis = linspace(-1,1, Nfft_a);
P = zeros(1, Nfft_a);
m = (0:M-1).';
for i = 1:Nfft_a
    a = exp(1j*2*pi*(m*d/lambda)*mu_axis(i));
    denom = real(a'*(En*En')*a);
    if denom <= 0, denom = eps; end
    P(i) = 1 / denom;
end
[~, ia] = max(P);
ang_hat = asind(mu_axis(ia));
end
