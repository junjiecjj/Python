function [hat_r_tars, hat_v_tars] = est_rv_clean2_z(z_time, pfa, T_prbs, num_of_targets, fs, lambda)
    % CLEAN-based Range–Doppler with UNIQUE range enforcement.
    % Input:
    %   z_time : [N_chip x N_sym], time-domain matched-filter output
    %   noise_power       : (not used here; kept for signature compatibility)
    %   pfa               : desired false alarm probability (e.g., 1e-6)
    %   T_prbs            : PRI (s) -> PRF = 1/T_prbs
    %   num_of_targets    : number of detections to return
    %   fs                : fast-time sampling rate (Hz)
    %   lambda            : wavelength (m)
    % Output:
    %   hat_r_tars        : [Kx1] estimated ranges (m)
    %   hat_v_tars        : [Kx1] estimated velocities (m/s)

    % ---------- Axes ----------
    c = physconst('Lightspeed');
    [Nr, Nd] = size(z_time);
    fasttime  = (0:Nr-1).' / fs;
    rangebins = c * fasttime / 2;
    PRF = 1 / T_prbs;

    % Doppler frequency axis
    if mod(Nd,2)==0
        fD = (-Nd/2:Nd/2-1) * (PRF/Nd);
    else
        fD = (-(Nd-1)/2:(Nd-1)/2) * (PRF/Nd);
    end

    % ---------- Pre-process ----------
    % Remove slow-time DC/clutter
    X = z_time - mean(z_time, 2);
    % Window across slow-time to reduce Doppler leakage
    wd = hann(Nd).';
    Xw = X .* wd;

    % ---------- RD map ----------
    RD = fftshift(fft(Xw, [], 2), 2);   % complex RD
    RDpow = abs(RD).^2;

    % ---------- Global detection threshold (robust) ----------
    noise_est = median(RDpow(:)) / log(2);  % robust noise power proxy
    M = Nd;                                   % effective looks
    thresh_dB = npwgnthresh(pfa, M, 'noncoherent');
    abs_thresh = noise_est * db2pow(thresh_dB);

    % ---------- Adaptive range guard from matched-filter mainlobe ----------
    % Measure mainlobe width (in bins) from the noncoherent range profile (pre-window).
    range_profile = sum(abs(X).^2, 2);
    mpd_bins = estimate_mainlobe_bins(range_profile);   % half-height width in bins
    mpd_bins = max(3, ceil(1.25 * mpd_bins));           % add safety margin (25%)

    % ---------- CLEAN parameters ----------
    patch_r = 5;        % half-width in range (bins) for the PSF patch
    patch_d = 5;        % half-width in Doppler (bins) for the PSF patch
    gain    = 0.8;      % loop gain
    maxIters = 10 * num_of_targets;

    % ---------- CLEAN loop with UNIQUE range enforcement ----------
    Rres = RD;                   % residual (complex)
    blocked_rows = false(Nr,1);  % rows we will not consider anymore (±mpd guard)

    r_list = zeros(num_of_targets,1);
    d_list = zeros(num_of_targets,1);
    a_list = zeros(num_of_targets,1);
    k = 0; iter = 0;

    while (k < num_of_targets) && (iter < maxIters)
        iter = iter + 1;

        % Find strongest residual outside blocked rows
        P = abs(Rres).^2;
        P(blocked_rows, :) = -Inf;      % mask out blocked ranges from consideration
        [val, idx] = max(P(:));
        if ~isfinite(val) || val < abs_thresh
            break;  % nothing strong enough remains
        end
        [ri, di] = ind2sub(size(P), idx);
        a = Rres(ri, di);               % complex amp at the peak

        % ----- Extract local PSF template from ORIGINAL RD (stabilizes shape) -----
        r1 = max(1, ri - patch_r); r2 = min(Nr, ri + patch_r);
        d1 = max(1, di - patch_d); d2 = min(Nd, di + patch_d);
        PSFpatch = RD(r1:r2, d1:d2);
        PSFpeak  = max(abs(PSFpatch(:)));
        if PSFpeak > 0
            PSFpatch = PSFpatch / (PSFpeak + eps);
        end

        % Build a zero template T with the patch at its actual location
        T = zeros(Nr, Nd);
        T(r1:r2, d1:d2) = PSFpatch;

        % ----- Subtract scaled template from residual -----
        Rres = Rres - gain * a * T;

        % ----- Record detection -----
        k = k + 1;
        r_list(k) = ri;
        d_list(k) = di;
        a_list(k) = a;

        % ----- Enforce UNIQUE range: block ±mpd_bins around ri -----
        L = max(1, ri - mpd_bins); R = min(Nr, ri + mpd_bins);
        blocked_rows(L:R) = true;
    end
    % ---------- If not enough detections, supplement with non-overlapping ranges ----------
    if k < num_of_targets
        % Find strong, non-overlapping peaks from range_profile
        [~, locs] = findpeaks(range_profile, 'SortStr','descend', 'MinPeakDistance', mpd_bins);
        for i = 1:numel(locs)
            if k >= num_of_targets, break; end
            r = locs(i);
            if any(blocked_rows(max(1,r-mpd_bins):min(Nr,r+mpd_bins)))
                continue; % overlaps, skip
            end
            % Choose strongest Doppler in this row from ORIGINAL RD
            [~, d] = max(RDpow(r,:));
            k = k + 1;
            r_list(k) = r;
            d_list(k) = d;
            a_list(k) = RD(r,d);
            % Block its range neighborhood
            L = max(1, r - mpd_bins); Rg = min(Nr, r + mpd_bins);
            blocked_rows(L:Rg) = true;
        end
    end

    % ---------- Outputs ----------
    hat_r_tars = zeros(num_of_targets,1);
    hat_v_tars = zeros(num_of_targets,1);

    if k > 0
        for i = 1:k
            hat_r_tars(i) = rangebins(r_list(i));
            hat_v_tars(i) = dop2speed(-fD(d_list(i)) / 4, lambda);
        end
    end
    % If k < num_of_targets, the remaining entries stay 0 by design.

end

% ===================== Local helper =====================

function mpd_bins = estimate_mainlobe_bins(range_profile)
    % Estimate the matched-filter mainlobe width (in bins) at half-height
    % from the strongest peak. Fallback to 3 bins if measurement fails.
    try
        [~,~,w] = findpeaks(range_profile, 'SortStr','descend', ...
                            'NPeaks',1, 'WidthReference','halfheight');
        if isempty(w) || ~isfinite(w(1))
            mpd_bins = 3;
        else
            mpd_bins = max(3, ceil(w(1)));
        end
    catch
        mpd_bins = 3;
    end
end

