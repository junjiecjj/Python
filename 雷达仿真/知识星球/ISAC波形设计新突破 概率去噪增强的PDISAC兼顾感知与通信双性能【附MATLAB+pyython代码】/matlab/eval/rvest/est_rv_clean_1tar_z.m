function [hat_r_tars, hat_v_tars] = est_rv_clean_1tar_z(z_time, pfa, T_prbs, num_of_targets, fs, lambda)

    c = physconst('Lightspeed');
    [Nr, Nd] = size(z_time);
    fasttime  = (0:Nr-1).' / fs;
    rangebins = c * fasttime / 2;
    PRF = 1 / T_prbs;

    % Pre-process
    X = z_time - mean(z_time, 2);
    wd = hann(Nd).';
    Xw = X .* wd;

    % Complex RD
    RD = fftshift(fft(Xw, [], 2), 2);       % complex
    RDpow = abs(RD).^2;

    % Use a global threshold as a stopping guard (robust median estimate)
    noise_est = median(RDpow(:)) / log(2);  % rough
    M = Nd;                                  % effective looks
    thresh_dB = npwgnthresh(pfa, M, 'noncoherent');
    abs_thresh = noise_est * db2pow(thresh_dB);

    % CLEAN parameters
    patch_r = 5; patch_d = 5;    % half-widths of PSF patch (total size ~ (2*+1))
    gain = 0.8;                  % loop gain
    maxIters = 10 * num_of_targets;

    Rres = RD;  % residual (complex)
    detections = []; %#ok<NASGU>
    r_list = zeros(num_of_targets,1);
    d_list = zeros(num_of_targets,1);
    a_list = zeros(num_of_targets,1);
    k = 0; iter = 0;

    while (k < num_of_targets) && (iter < maxIters)
        iter = iter + 1;
        [val, idx] = max(abs(Rres(:)).^2);
        if isempty(idx) || val < abs_thresh
            break;
        end
        [ri, di] = ind2sub(size(Rres), idx);
        a = Rres(ri, di);  % complex amp at peak

        % Extract PSF patch around (ri,di) from the ORIGINAL RD (stabilizes shape)
        r1 = max(1, ri-patch_r); r2 = min(Nr, ri+patch_r);
        d1 = max(1, di-patch_d); d2 = min(Nd, di+patch_d);
        PSFpatch = RD(r1:r2, d1:d2);
        % Normalize template to unit peak magnitude
        PSFpatch = PSFpatch / max(abs(PSFpatch(:)) + eps);

        % Build a zero template and place the patch at (ri,di)
        T = zeros(Nr, Nd);
        T(r1:r2, d1:d2) = PSFpatch;

        % Subtract scaled, shifted template from residual
        Rres = Rres - gain * a * T;

        % Record detection
        k = k + 1;
        r_list(k) = ri;
        d_list(k) = di;
        a_list(k) = a;
    end

    % Map to outputs
    hat_r_tars = zeros(num_of_targets,1);
    hat_v_tars = zeros(num_of_targets,1);

    % Doppler axis
    if mod(Nd,2)==0
        fD = (-Nd/2:Nd/2-1) * (PRF/Nd) / 2;
    else
        fD = (-(Nd-1)/2:(Nd-1)/2) * (PRF/Nd) / 2;
    end

    for i = 1:k
        hat_r_tars(i) = rangebins(r_list(i));
        hat_v_tars(i) = dop2speed(-fD(d_list(i)) / 2, lambda);
    end

end

