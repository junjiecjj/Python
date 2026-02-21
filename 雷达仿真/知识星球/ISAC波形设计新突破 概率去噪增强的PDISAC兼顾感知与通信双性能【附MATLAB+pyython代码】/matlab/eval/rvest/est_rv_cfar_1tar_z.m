function [hat_r_tars, hat_v_tars] = est_rv_cfar_1tar_z(z_time, pfa, T_prbs, num_of_targets, fs, lambda)

    % Inputs and outputs match your original expectations
    % z_time: [N_chip x N_sym], already matched-filter output

    c = physconst('Lightspeed');
    [Nr, Nd] = size(z_time);

    % Axes
    fasttime  = (0:Nr-1).' / fs;
    rangebins = c * fasttime / 2;
    PRF = 1 / T_prbs;

    % Pre-process: suppress DC/clutter across slow time
    X = z_time - mean(z_time, 2);   % remove slow-time mean

    % Window across slow-time to reduce Doppler leakage
    wd = hann(Nd).';
    Xw = X .* wd;  % broadcasting along columns

    % Doppler FFT (centered)
    RD = fftshift(fft(Xw, [], 2), 2);
    RDpow = abs(RD).^2;

    % --- 2D CA-CFAR detection on RDpow ---
    % Parameters (tune for your scenario)
    guard_r = 1;  train_r = 8;   % range guard/training cells (bins)
    guard_d = 1;  train_d = 8;   % Doppler guard/training cells
    alpha = cfar_alpha_2d(train_r, train_d, pfa);  % scale factor from Pfa

    det_mask = cfar2d_ca(RDpow, train_r, guard_r, train_d, guard_d, alpha);

    % Suppress detections near edges where CFAR is invalid
    det_mask([1:train_r+guard_r, end-(train_r+guard_r-1):end], :) = false;
    det_mask(:, [1:train_d+guard_d, end-(train_d+guard_d-1):end]) = false;

    % Pick the strongest K detections
    RDcand = RDpow .* det_mask;
    [~, linIdx] = maxk(RDcand(:), num_of_targets);

    % Prepare outputs
    hat_r_tars = zeros(num_of_targets,1);
    hat_v_tars = zeros(num_of_targets,1);

    if ~isempty(linIdx)
        [r_idx, d_idx] = ind2sub(size(RDpow), linIdx);

        % Doppler frequency axis
        if mod(Nd,2)==0
            fD = (-Nd/2:Nd/2-1) * (PRF/Nd);
        else
            fD = (-(Nd-1)/2:(Nd-1)/2) * (PRF/Nd);
        end

        for k = 1:numel(linIdx)
            hat_r_tars(k) = rangebins(r_idx(k));
            fd = fD(d_idx(k))/2;
            hat_v_tars(k) = dop2speed(-fd / 2, lambda);
        end
    end
end

% ---- helpers ----

function alpha = cfar_alpha_2d(Tr, Td, pfa)
    % Scale factor alpha for 2D CA-CFAR with Tr x 2 and Td x 2 training cells.
    % Total training cells:
    N = 2*Tr*(2*Td + 2*Td) + 2*Td*(2*Tr) + 4*Tr*Td;  %#ok<NASGU> (kept for clarity)

    % Correct count: training cells are a ring around guard+CUT:
    Nr = 2*Tr + 2*Td + 4*Tr*Td; % This naive formula isn't right; compute explicitly below.

    % Build exact 2D stencil to count training cells:
    % We'll compute alpha via the standard CA-CFAR formula:
    % alpha = N * (pfa^(-1/N) - 1)
    N = (2*Tr + 2*Td + 4*Tr*Td); % placeholder; we compute exact below
    % Let's compute exact N for given Tr,Td (excluding guards and CUT) by simulation of a mask
    r = Tr; g = 0; d = Td; % (we only need counts; guards not part of training)
    Mr = 2*r + 1; Md = 2*d + 1;
    mask = true(Mr, Md);
    mask(r+1, d+1) = false; % CUT not included
    N = nnz(mask);
    alpha = N * (pfa^(-1/N) - 1);
end

function det = cfar2d_ca(P, Tr, Gr, Td, Gd, alpha)
    % Simple 2D CA-CFAR on power map P (non-negative).
    [nr, nd] = size(P);
    det = false(nr, nd);
    for i = (Tr+Gr+1):(nr-(Tr+Gr))
        for j = (Td+Gd+1):(nd-(Td+Gd))
            r1 = i-(Tr+Gr); r2 = i+(Tr+Gr);
            d1 = j-(Td+Gd); d2 = j+(Td+Gd);
            window = P(r1:r2, d1:d2);

            % Exclude guard band + CUT
            gwin = false(size(window));
            gwin(Tr+1:Tr+2*Gr+1, Td+1:Td+2*Gd+1) = true; % guard+CUT
            training = window(~gwin);
            mu = mean(training(:));

            th = alpha * mu;
            det(i,j) = P(i,j) > th;
        end
    end
end