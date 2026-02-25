function [hat_r_tars, hat_v_tars] = est_rv_music_1tar_z(z_time, pfa, T_slow, num_of_targets, fs, lambda)

    c = physconst('Lightspeed');
    [Nr, Nd] = size(z_time);
    fasttime  = (0:Nr-1).' / fs;
    rangebins = c * fasttime / 2;
    PRF = 1 / T_slow;

    % 1) Range detection (noncoherent)
    X = z_time;
    range_profile = sum(abs(X).^2, 2);

    train = 16; guard = 2;
    alpha = cfar_alpha_1d(train, pfa);
    det_r = cfar1d_ca(range_profile, train, guard, alpha);
    r_cand = find(det_r);

    if isempty(r_cand)
        [~, r_cand] = maxk(range_profile, min(num_of_targets, Nr));
    end

    % Limit to at most num_of_targets range bins
    r_cand = r_cand(1:min(num_of_targets, numel(r_cand)));

    % 2) Doppler MUSIC per detected range
    % MUSIC settings
    L = min( max(8, round(Nd/3)), Nd-2 ); % subarray length
    Kmodel = 1;                           % model order per range bin (1 Doppler per target)
    fd_grid = linspace(-PRF/2, PRF/2, 1024);

    hat_r_tars = zeros(num_of_targets,1);
    hat_v_tars = zeros(num_of_targets,1);

    for k = 1:numel(r_cand)
        r = r_cand(k);
        x = X(r, :).';
        x = x - mean(x); % remove DC/clutter

        % Hankel embedding for spatial smoothing
        Ksnap = Nd - L + 1;
        H = zeros(L, Ksnap);
        for s = 1:Ksnap
            H(:, s) = x(s:s+L-1);
        end

        % Covariance with diagonal loading
        R = (H*H')/Ksnap;
        R = (R + R')/2;
        dl = 1e-3 * trace(R)/L;
        R = R + dl*eye(L);

        % Eigendecomposition
        [U, D] = eig(R);
        [~, idx] = sort(diag(D), 'ascend');
        Un = U(:, idx(1:end-Kmodel));  % noise subspace

        % MUSIC pseudospectrum across fd
        m = (0:L-1).';
        P = zeros(size(fd_grid));
        for ii = 1:numel(fd_grid)
            a = exp(1j*2*pi*fd_grid(ii)*T_slow*m);
            denom = real(a'*(Un*Un')*a) + 1e-12;
            P(ii) = 1./denom;
        end
        [~, imax] = max(P);
        fd_hat = fd_grid(imax) / 2;

        hat_r_tars(k) = rangebins(r);
        hat_v_tars(k) = dop2speed(-fd_hat/2, lambda);
    end
    % remaining entries stay zero if fewer than requested
end


function alpha = cfar_alpha_1d(Tr, pfa)
    N = 2*Tr;
    alpha = N * (pfa^(-1/N) - 1);
end

function det = cfar1d_ca(x, Tr, Gr, alpha)
    N = numel(x);
    det = false(N,1);
    for i = (Tr+Gr+1):(N-(Tr+Gr))
        idxL = i-(Tr+Gr):i-(Gr+1);
        idxR = i+(Gr+1):i+(Tr+Gr);
        mu = mean([x(idxL); x(idxR)]);
        th = alpha * mu;
        det(i) = x(i) > th;
    end
end