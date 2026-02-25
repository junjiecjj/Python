function [hat_r_tars, hat_v_tars] = est_rv_music2_z(z_time, pfa, T_prbs, num_of_targets, fs, lambda)
    % RANGE-FIRST + DOPPLER MUSIC per range with UNIQUE range enforcement.
    % Input:
    %   z_time : [N_chip x N_sym], time-domain matched-filter output (z_PRBS_waveform)
    %   noise_power       : (kept for signature compatibility; not used)
    %   pfa               : desired Pfa for range CFAR (e.g., 1e-6)
    %   T_prbs            : PRI (s) -> PRF = 1/T_prbs
    %   num_of_targets    : number of (range, velocity) pairs to return
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

    % ---------- 1) Noncoherent range profile ----------
    % Use matched-filter output directly (no extra range compression).
    X = z_time;
    range_profile = sum(abs(X).^2, 2);   % power sum across slow-time

    % ---------- 2) Range CFAR (1D) to get candidates ----------
    Tr = 16;               % training bins per side (tune as needed)
    Gr = 2;                % guard bins per side (tune as needed)
    alpha = local_cfar_alpha_1d(Tr, pfa);
    det_r = local_cfar1d_ca(range_profile, Tr, Gr, alpha);
    cand = find(det_r);

    % ---------- 3) Adaptive range guard using matched-filter mainlobe width ----------
    % Measure half-height width of strongest range mainlobe (in bins).
    mpd_bins = local_estimate_mainlobe_bins(range_profile);
    mpd_bins = max(3, ceil(1.25 * mpd_bins));  % +25% safety margin

    % ---------- 4) Fallbacks if CFAR sparse ----------
    if isempty(cand)
        % Peak-pick strongest, enforcing min distance ~ mainlobe width
        [~, cand] = findpeaks(range_profile, 'SortStr','descend', ...
                              'MinPeakDistance', mpd_bins);
    end
    if isempty(cand)
        % Take strongest bins by power
        [~, cand] = maxk(range_profile, min(num_of_targets, Nr));
    end
    % ---------- 5) Enforce UNIQUE ranges (non-overlapping) ----------
    r_sel = local_greedy_unique(cand, range_profile, mpd_bins, num_of_targets);
    K = numel(r_sel);

    % ---------- 6) MUSIC Doppler per selected range ----------
    hat_r_tars = zeros(num_of_targets,1);
    hat_v_tars = zeros(num_of_targets,1);

    % MUSIC settings
    % Subarray length L (Hankel). Keep 8 <= L <= Nd-2
    L = min(max(8, round(Nd/3)), Nd-2);
    L = max(3, min(L, Nd-2));   % safety
    Kmodel = 1;                 % 1 Doppler tone per selected range
    fd_grid = linspace(-PRF/2, PRF/2, 2048);

    % Remove slow-time DC (clutter) before covariance
    X0 = X - mean(X, 2);

    for k = 1:K
        r = r_sel(k);
        x = X0(r, :).';              % slow-time cut (column vector)

        % Hankel embedding for spatial smoothing
        Ksnap = Nd - L + 1;
        if Ksnap < 2
            % Degenerate case: fall back to FFT if too few snapshots
            fd = fallback_fft_doppler(x, PRF);
        else
            H = zeros(L, Ksnap);
            for s = 1:Ksnap
                H(:, s) = x(s:s+L-1);
            end

            % Covariance with diagonal loading
            R = (H*H')/Ksnap;
            R = (R + R')/2;
            dl = 1e-3 * trace(R)/L;
            R = R + dl*eye(L);

            % Optional: Forward-backward averaging (often helps for complex sinusoids)
            % J = flipud(eye(L)); R = (R + J*conj(R)*J) / 2;

            % Eigendecomposition
            [U, D] = eig(R);
            [~, idx] = sort(diag(D), 'ascend');
            Un = U(:, idx(1:end-Kmodel));   % noise subspace

            % MUSIC pseudospectrum over fd_grid
            m = (0:L-1).';
            P = zeros(size(fd_grid));
            UnUnH = Un*Un';
            for ii = 1:numel(fd_grid)
                a = exp(1j*2*pi*fd_grid(ii)*T_prbs*m);
                denom = real(a'*(UnUnH)*a) + 1e-12;
                P(ii) = 1/denom;
            end
            [~, imax] = max(P);
            fd = fd_grid(imax);
        end
        hat_r_tars(k) = rangebins(r);
        hat_v_tars(k) = dop2speed(-fd / 4, lambda);
    end

    % ---------- 7) If fewer than requested, supplement with non-overlapping ranges ----------
    if K < num_of_targets
        % Block already selected ranges
        taken = false(Nr,1);
        for ii = 1:K
            r = r_sel(ii);
            Lr = max(1, r - mpd_bins); Rr = min(Nr, r + mpd_bins);
            taken(Lr:Rr) = true;
        end

        % Additional non-overlapping peaks from range profile
        [~, locs] = findpeaks(range_profile, 'SortStr','descend', 'MinPeakDistance', mpd_bins);
        for i = 1:numel(locs)
            if sum(hat_r_tars ~= 0) >= num_of_targets, break; end
            r = locs(i);
            if any(taken(max(1,r-mpd_bins):min(Nr,r+mpd_bins))), continue; end

            % Doppler via MUSIC for this row (or FFT fallback if degenerate)
            x = X0(r, :).';
            Ksnap = Nd - L + 1;
            if Ksnap < 2
                fd = fallback_fft_doppler(x, PRF);
            else
                H = zeros(L, Ksnap);
                for s = 1:Ksnap
                    H(:, s) = x(s:s+L-1);
                end
                R = (H*H')/Ksnap; R = (R + R')/2;
                dl = 1e-3 * trace(R)/L; R = R + dl*eye(L);
                [U, D] = eig(R);
                [~, idx] = sort(diag(D), 'ascend');
                Un = U(:, idx(1:end-Kmodel));
                m = (0:L-1).';
                P = zeros(size(fd_grid));
                UnUnH = Un*Un';
                for ii = 1:numel(fd_grid)
                    a = exp(1j*2*pi*fd_grid(ii)*T_prbs*m);
                    denom = real(a'*(UnUnH)*a) + 1e-12;
                    P(ii) = 1/denom;
                end
                [~, imax] = max(P);
                fd = fd_grid(imax);
            end

            % Write into first empty slot
            pos = find(hat_r_tars==0, 1, 'first');
            hat_r_tars(pos) = rangebins(r);
            hat_v_tars(pos) = dop2speed(-fd / 4, lambda);

            % Block this range neighborhood
            Lr = max(1, r - mpd_bins); Rr = min(Nr, r + mpd_bins);
            taken(Lr:Rr) = true;
        end
    end
end


% ---- local helpers (scoped) ----
function alpha = local_cfar_alpha_1d(Tr, pfa_)
    N_ = 2*Tr;
    alpha = N_ * (pfa_^(-1/N_) - 1);
end
function det = local_cfar1d_ca(x_, Tr, Gr, alpha_)
    N_ = numel(x_); det = false(N_,1);
    for ii_ = (Tr+Gr+1):(N_-(Tr+Gr))
        idxL = ii_-(Tr+Gr):ii_-(Gr+1);
        idxR = ii_+(Gr+1):ii_+(Tr+Gr);
        mu_  = mean([x_(idxL); x_(idxR)]);
        det(ii_) = x_(ii_) > alpha_*mu_;
    end
end

function mpd_bins_ = local_estimate_mainlobe_bins(x_)
    try
        [~,~,w_] = findpeaks(x_, 'SortStr','descend', 'NPeaks',1, 'WidthReference','halfheight');
        mpd_bins_ = max(3, ceil(w_(1)));
    catch
        mpd_bins_ = 3;
    end
end

function sel = local_greedy_unique(candidates, score, guard_bins, Ksel)
    candidates = candidates(:).';
    [~, ord] = sort(score(candidates), 'descend');
    candidates = candidates(ord);
    taken = false(numel(score),1);
    sel = [];
    for jj = 1:numel(candidates)
        idx = candidates(jj);
        Lr = max(1, idx-guard_bins); Rr = min(numel(score), idx+guard_bins);
        if ~any(taken(Lr:Rr))
            sel(end+1) = idx; %#ok<AGROW>
            taken(Lr:Rr) = true;
            if numel(sel) >= Ksel, break; end
        end
    end
    sel = sel(:);
end

function fd_ = fallback_fft_doppler(ts, PRF_)
    % Fallback if MUSIC covariance is ill-conditioned (too few snapshots)
    Nd_ = numel(ts);
    Nfft_ = 2^nextpow2(max(256, 4*Nd_));
    w_ = hann(Nd_);
    fda_ = (-Nfft_/2:Nfft_/2-1) * (PRF_/Nfft_);
    Sd_ = fftshift(fft((ts - mean(ts)) .* w_, Nfft_));
    [~, imax_] = max(abs(Sd_).^2);
    fd_ = fda_(imax_);
end
