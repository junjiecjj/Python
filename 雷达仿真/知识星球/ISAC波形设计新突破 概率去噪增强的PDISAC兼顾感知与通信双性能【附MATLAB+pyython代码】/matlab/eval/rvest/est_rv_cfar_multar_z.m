function [hat_r_tars, hat_v_tars] = est_rv_cfar3_z(z_time, pfa, T_prbs, num_of_targets, fs, lambda)
    % 2D-FFT Range–Doppler with UNIQUE range enforcement.
    % Input:
    %   z_time : [N_chip x N_sym], matched-filter output in TIME domain
    %   noise_power       : (kept for signature compatibility; not used here)
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

    % Doppler frequency axis (no zero padding for RD map)
    if mod(Nd,2)==0
        fD = (-Nd/2:Nd/2-1) * (PRF/Nd);
    else
        fD = (-(Nd-1)/2:(Nd-1)/2) * (PRF/Nd);
    end

    % ---------- Pre-process ----------
    % Remove slow-time DC per range bin (clutter)
    X = z_time - mean(z_time, 2);
    % Window across slow-time to reduce Doppler leakage
    wd = hann(Nd).';
    Xw = X .* wd;  % implicit expansion over columns

    % ---------- RD map ----------
    RD = fftshift(fft(Xw, [], 2), 2);  % complex RD
    RDpow = abs(RD).^2;

    % ---------- Adaptive range guard (mainlobe width) ----------
    % Estimate range mainlobe width from noncoherent range profile
    range_profile = sum(abs(X).^2, 2);  % power sum across slow-time (before window)
    mpd_bins = estimate_mainlobe_bins(range_profile);   % bins (half-height width)
    mpd_bins = max(3, ceil(1.25 * mpd_bins));           % safety factor (25%)

    % ---------- 2D CA-CFAR ----------
    % Tune these to your scene
    train_r = 8;  guard_r = 1;     % range training/guard (bins)
    train_d = 8;  guard_d = 1;     % Doppler training/guard (bins)

    alpha = cfar_alpha_2d(train_r, guard_r, train_d, guard_d, pfa);
    det_mask = cfar2d_ca(RDpow, train_r, guard_r, train_d, guard_d, alpha);

    % Suppress borders where CFAR not valid
    det_mask([1:train_r+guard_r, end-(train_r+guard_r-1):end], :) = false;
    det_mask(:, [1:train_d+guard_d, end-(train_d+guard_d-1):end]) = false;

    % ---------- Build candidate list from CFAR hits ----------
    [cand_r, cand_d] = find(det_mask);
    cand_pow = RDpow(det_mask);
    % If no CFAR hits, fall back to the entire map
    if isempty(cand_pow)
        [cand_pow, linIdx] = maxk(RDpow(:), min(10*num_of_targets, numel(RDpow)));
        [cand_r, cand_d] = ind2sub(size(RDpow), linIdx);
    end
    % Sort by descending power
    [~, ord] = sort(cand_pow, 'descend');
    cand_r = cand_r(ord); cand_d = cand_d(ord);

    % ---------- Greedy selection with UNIQUE range constraint ----------
    sel_r = []; sel_d = [];
    taken = false(Nr,1);
    for n = 1:numel(cand_r)
        r = cand_r(n);
        % Reject if overlaps in range with any chosen detection
        if ~any(taken(max(1,r-mpd_bins):min(Nr, r+mpd_bins)))
            sel_r(end+1,1) = r; %#ok<AGROW>
            sel_d(end+1,1) = cand_d(n);
            % Mark guard in range to avoid overlap
            L = max(1, r-mpd_bins); R = min(Nr, r+mpd_bins);
            taken(L:R) = true;
            if numel(sel_r) >= num_of_targets, break; end
        end
    end

    % ---------- If not enough unique ranges, supplement from 1D range profile ----------
    if numel(sel_r) < num_of_targets
        % Find strong, non-overlapping peaks in range_profile
        [~, locs] = findpeaks(range_profile, 'SortStr','descend', 'MinPeakDistance', mpd_bins);
        for i = 1:numel(locs)
            r = locs(i);
            if ~any(taken(max(1,r-mpd_bins):min(Nr, r+mpd_bins)))
                % Pick strongest Doppler at this range row
                [~, d] = max(RDpow(r,:));
                sel_r(end+1,1) = r; %#ok<AGROW>
                sel_d(end+1,1) = d;
                L = max(1, r-mpd_bins); R = min(Nr, r+mpd_bins);
                taken(L:R) = true;
                if numel(sel_r) >= num_of_targets, break; end
            end
        end
    end

    % ---------- Prepare outputs ----------
    hat_r_tars = zeros(num_of_targets,1);
    hat_v_tars = zeros(num_of_targets,1);

    K = min(numel(sel_r), num_of_targets);
    for k = 1:K
        hat_r_tars(k) = rangebins(sel_r(k));
        hat_v_tars(k) = dop2speed(-fD(sel_d(k))/4, lambda);
    end
    % remaining entries stay zero if fewer than requested

end

% ===================== Helpers (scoped to this file) =====================

function mpd_bins = estimate_mainlobe_bins(range_profile)
    % Estimate range mainlobe width (in bins) from strongest peak (half-height).
    try
        [~,~,w] = findpeaks(range_profile, 'SortStr','descend', ...
                            'NPeaks',1, 'WidthReference','halfheight');
        if isempty(w) || ~isfinite(w(1)), mpd_bins = 3; else, mpd_bins = max(3, ceil(w(1))); end
    catch
        mpd_bins = 3;
    end
    end

    function alpha = cfar_alpha_2d(Tr, Gr, Td, Gd, pfa)
    % Proper 2D CA-CFAR scale factor based on training cell count.
    % Window size around CUT: (2*(Tr+Gr)+1) x (2*(Td+Gd)+1)
    % Guard+CUT size        : (2*Gr+1)      x (2*Gd+1)
    W_r = 2*(Tr+Gr) + 1;   W_d = 2*(Td+Gd) + 1;
    G_r = 2*Gr + 1;        G_d = 2*Gd + 1;
    Ntrain = W_r*W_d - G_r*G_d;   % exclude guard region incl. CUT
    alpha  = Ntrain * (pfa^(-1/Ntrain) - 1);
end

function det = cfar2d_ca(P, Tr, Gr, Td, Gd, alpha)
    % 2D CA-CFAR (cell-averaging) on power map P (non-negative).
    [nr, nd] = size(P);
    det = false(nr, nd);

    W_r = 2*(Tr+Gr) + 1;  W_d = 2*(Td+Gd) + 1;
    % Valid index ranges (avoid border where full window doesn't fit)
    ir1 = Tr+Gr+1; ir2 = nr - (Tr+Gr);
    id1 = Td+Gd+1; id2 = nd - (Td+Gd);

    for i = ir1:ir2
        for j = id1:id2
            r1 = i-(Tr+Gr); r2 = i+(Tr+Gr);
            d1 = j-(Td+Gd); d2 = j+(Td+Gd);
            window = P(r1:r2, d1:d2);

            % Guard region (including CUT): centered at (i,j)
            center_r = Tr+Gr+1; center_d = Td+Gd+1;
            gmask = false(W_r, W_d);
            rL = center_r - Gd_to_Gr(Gr); rR = center_r + Gd_to_Gr(Gr);
            dL = center_d - Gd_to_Gr(Gd); dR = center_d + Gd_to_Gr(Gd);
            gmask(rL:rR, dL:dR) = true;

            train = window(~gmask);
            mu = mean(train(:));
            th = alpha * mu;
            det(i,j) = P(i,j) > th;
        end
    end
end

function v = Gd_to_Gr(x)
    % tiny helper to keep expressions clean
    v = x;
end