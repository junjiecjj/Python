function [tau_norm, nu_norm, AFdB_zerodop, AFdB_zerodel, metrics_delay, metrics_dopp] = AF_fullAnalysis(s, params, OS_del, OS_dop)
    %
    % =========================================================================
    % Author    : Dr. (Eric) Hyeon Seok Rou
    % Version   : v1.0
    % Date      : Oct 13, 2025
    % =========================================================================

    addpath('helperFunctions/');
    % Compute AF cuts
    [tau_norm, AFdB_zerodop, AF_zerodop] = AF_zerodopplercut(s, OS_del, 4);
    [nu_norm, AFdB_zerodel, AF_zerodel] = AF_zerodelaycut(s, OS_dop, 1);
    lenDopGuard = floor(length(nu_norm)/20);
    nu_norm = nu_norm(lenDopGuard+1:end-lenDopGuard);
    AFdB_zerodel = AFdB_zerodel(lenDopGuard+1:end-lenDopGuard);
    AF_zerodel = AF_zerodel(lenDopGuard+1:end-lenDopGuard);

    % Delay cut metrics
    metrics_delay = compute_delay_metrics(tau_norm, AF_zerodop, params);
    % Doppler cut metrics
    metrics_dopp  = compute_doppler_metrics(nu_norm, AF_zerodel, params);
end

function [tau_norm, cut_dB, cut] = AF_zerodopplercut(s, OS_delay, Lh)
% Computes the zero-Doppler cut A(τ,0) with optional fractional-delay
% Aperiodic (zero-padded overlap) model, built from first principles.
%
% INPUTS
%   s          : length-N complex/real vector (row or column)
%   Ts         : sampling period [s] (default 1)   -- used only for tau_phys (not returned)
%   OS_delay   : delay oversampling factor (default 1 => integer delays only)
%   Lh         : half-length of windowed-sinc kernel 
%
% OUTPUTS
%   tau_norm   : normalized delay grid (tau/N), dimensionless, centered at 0
%   cut_dB     : |A(tau,0)| in dB, normalized to 0 dB peak
%   cut        : complex A(tau,0)
% =========================================================================
% Author    : Dr. (Eric) Hyeon Seok Rou
% Version   : v1.0
% Date      : Oct 13, 2025
% =========================================================================

    % ----- defaults -----
    if OS_delay < 1, 
        OS_delay = 1; 
    end
    if Lh < 1, 
        Lh = 1; 
    end

    % ----- validate & prep -----
    s = double(s(:));
    N = length(s);

    % energy Es = sum |s[n]|^2
    Es = 0.0;
    for n = 1:N
        sr = real(s(n)); si = imag(s(n));
        Es = Es + (sr*sr + si*si);
    end

    % ----- fractional delay grid -----
    tau_int  = (-(N-1):(N-1));                 % integer endpoints
    tau_frac = (tau_int(1) : 1/OS_delay : tau_int(end)).';  % step 1/OS_delay
    M = length(tau_frac);
    cut = complex(zeros(M,1));

    % ----- smooth compact window, |x|<=Lh -----
    % w(x) = (1 - (|x|/Lh)^2)^2 for |x|<=Lh; 0 otherwise. Ensures w(0)=1.
    function w = winCompact(x)
        ax = abs(x);
        w = zeros(size(x));
        for i = 1:numel(x)
            if ax(i) <= Lh
                u = ax(i)/Lh;
                w(i) = (1 - u*u)^2;
            else
                w(i) = 0.0;
            end
        end
    end

    % ----- main computation: A(tau,0) with fractional reconstruction of s[n-τ] -----
    for idx = 1:M
        tau_m = tau_frac(idx);     % possibly fractional delay

        % Sum over n where interpolation footprint overlaps data region.
        accR = 0.0; accI = 0.0;

        for n = 1:N
            % Reconstruct s[n - tau_m] by finite windowed-sinc from data samples s[m]
            xcenter = (n - tau_m);

            % Limit m to kernel support around xcenter
            mMin = max(1, ceil(xcenter - Lh));
            mMax = min(N, floor(xcenter + Lh));

            srecR = 0.0; srecI = 0.0;
            for m = mMin:mMax
                dx = xcenter - m;      % distance from grid sample m to fractional point
                % sinc(x) with sinc(0)=1
                if dx == 0
                    sincv = 1.0;
                else
                    sincv = sin(pi*dx) / (pi*dx);
                end
                wv = winCompact(dx);

                smR = real(s(m)); smI = imag(s(m));
                srecR = srecR + smR * (sincv * wv);
                srecI = srecI + smI * (sincv * wv);
            end

            % s[n] * conj( s_rec )
            sr = real(s(n)); si = imag(s(n));
            tr = srecR;       ti = srecI;
            prodR =  sr*tr + si*ti;
            prodI = -sr*ti + si*tr;

            accR = accR + prodR;
            accI = accI + prodI;
        end

        cut(idx) = complex(accR, accI);
    end

    % ----- normalization so A(0)=1 when Es>0-----
    if true && Es > 0
        invEs = 1.0/Es;
        for i = 1:M
            cut(i) = cut(i) * invEs;
        end
    end

    % ----- normalized delay axis -----
    tau_norm = tau_frac / N;   % dimensionless, centered

    % ----- dB output 0 dB at peak -----
    % magnitude
    peak = 0.0;
    mag  = zeros(M,1);
    for i = 1:M
        cr = real(cut(i)); ci = imag(cut(i));
        mi = sqrt(cr*cr + ci*ci);
        mag(i) = mi;
        if mi > peak, peak = mi; end
    end
    if peak == 0, peak = 1; end

    dB_floor = -inf;
    cut_dB = zeros(M,1);
    for i = 1:M
        val = mag(i) / peak;
        if val <= 0
            cut_dB(i) = dB_floor;
        else
            di = 20*log10(val);
            if di < dB_floor, di = dB_floor; end
            cut_dB(i) = di;
        end
    end
end

function [nu_os, AdB_os, A_os] = AF_zerodelaycut(s, OS, doppler_span)
    % Zero-delay cut A(0,nu) evaluated densely and exactly
    % INPUTS
    %   s            : length-N complex/real vector (row or column)
    %   OS           : Doppler oversampling factor (e.g., 32, 64, 128)
    %   doppler_span : plot span in normalized units; evaluate nu in [-span, +span]
    % OUTPUTS
    %   nu_dense  : 1×Nd vector of normalized Doppler points (includes 0 exactly)
    %   AdB_dense : 1×Nd zero-delay cut in dB, normalized (peak = 0 dB)
    %
    % =========================================================================
    % Author    : Dr. (Eric) Hyeon Seok Rou
    % Version   : v1.0
    % Date      : Oct 13, 2025
    % =========================================================================
    
    if nargin < 2 || isempty(OS), OS = 64; end
    if nargin < 3 || isempty(doppler_span), doppler_span = 5; end
    if OS < 1, OS = 1; end
    
    s = s(:);
    N = length(s);
    
    % Energy of s: Es = sum |s[n]|^2
    Es = 0.0;
    for n = 1:N
        sr = real(s(n)); si = imag(s(n));
        Es = Es + (sr*sr + si*si);
    end
    if Es == 0
        nu_os = linspace(-doppler_span, doppler_span, max(2, OS*N));
        AdB_os = dB_floor*ones(size(nu_os));
        return;
    end
    
    % Number of dense Doppler samples
    Nd = max(2, OS * N);
    
    % Build symmetric Doppler grid that GUARANTEES nu = 0 is on-grid
    if mod(Nd,2)==1
        nu_os = linspace(-doppler_span, +doppler_span, Nd);
    else
        Nd_eff   = Nd + 1;
        nu_full  = linspace(-doppler_span, +doppler_span, Nd_eff);
        nu_os = nu_full(1:Nd); % drop last endpoint (+span) so 0 is included
    end
    
    % A(0,nu) = sum_{n=0}^{N-1} |s[n]|^2 * exp(-j 2 pi nu n)
    A0 = complex(zeros(1, numel(nu_os)));
    for k = 1:numel(nu_os)
        nu_k = nu_os(k);
        accR = 0.0; accI = 0.0;
        for n = 1:N
            % |s[n]|^2
            sr = real(s(n)); si = imag(s(n));
            p  = sr*sr + si*si;
    
            % exp(-j 2 pi nu_k (n-1))  [use 0-based time index]
            phase = -2*pi*nu_k*(n-1);
            c = cos(phase); d = sin(phase);
    
            % accumulate p * (c + j d)
            accR = accR + p*c;
            accI = accI + p*d;
        end
        A0(k) = complex(accR, accI);
    end
    A_os = A0;
    A0abs = abs(A0);
    A0abs = A0abs ./ (max(A0abs) + eps);
    AdB_os = 20*log10(A0abs + eps);
end


function M = compute_delay_metrics(x_norm, A_cut, params)
    % x_norm   : tau_norm = tau / N  (dimensionless, centered)
    % A_cut    : complex A(tau,0)    (no dB floor)
    % params   : struct with fields:
    %            .N   = #samples
    %            .Fs  = sampling frequency [Hz]
    %            .B   = (optional) occupied bandwidth [Hz] (for reference limits)
    %            .Tsym= (optional) symbol time [s]         (for reporting only)
    %            .Tobs= (optional) observation time [s]; default N/Fs
    %
    % Returns: 3 dB width (normalized, samples, seconds), PSLR, ISLR, and references
    %
    %
    % =========================================================================
    % Author    : Dr. (Eric) Hyeon Seok Rou
    % Version   : v1.0
    % Date      : Oct 13, 2025
    % =========================================================================
    
    N   = params.N;
    Fs  = params.Fs;
    Ts  = 1 / Fs;
    if isfield(params,'Tobs') && ~isempty(params.Tobs)
        Tobs = params.Tobs;
    else
        Tobs = N / Fs;  % coherent time used by the AF
    end
    B = []; if isfield(params,'B'), B = params.B; end
    Tsym = []; if isfield(params,'Tsym'), Tsym = params.Tsym; end
    
    % ---- magnitude / power ----
    mag = abs(A_cut);
    if all(mag==0), error('Delay cut is all zeros.'); end
    mag = mag / max(mag);
    pwr = mag.^2;
    
    % ---- find mainlobe bounds ----
    [~, ipk] = max(mag);
    [iL, iR] = first_minima_around_peak(mag, ipk);
    if isempty(iL) || isempty(iR) || iL>=iR
        [iL, iR] = halfpower_bounds(mag, ipk);
    end
    
    % ---- 3 dB crossings (linear interp) ----
    xL = halfpower_crossing(x_norm, mag, ipk, -1);
    xR = halfpower_crossing(x_norm, mag, ipk, +1);
    width_norm   = xR - xL;        % in tau/N  (dimensionless)
    width_samp   = width_norm * N; % samples
    width_sec    = width_samp * Ts;
    
    % ---- PSLR / ISLR ----
    max_sl = max([peak_outside(mag, 1, iL-1), peak_outside(mag, iR+1, numel(mag))]);
    if isempty(max_sl) || isnan(max_sl), max_sl = 0; end
    PSLR_dB = 20*log10(max_sl + eps);
    
    P_main = sum(pwr(iL:iR));
    P_side = sum(pwr) - P_main;
    ISLR_dB = 10*log10( max(P_side,eps) / max(P_main,eps) );
    
    % ---- package ----
    M = struct();
    M.width_norm     = width_norm;          % tau/N 
    M.width_samples  = width_samp;          % samples
    M.width_seconds  = width_sec;           % seconds
    M.PSLR_dB        = PSLR_dB;
    M.ISLR_dB        = ISLR_dB;
    M.bounds_idx     = [iL iR];
    M.crossings_norm = [xL xR];
    M.scales         = struct('N',N,'Fs',Fs,'Ts',Ts,'Tobs',Tobs,'Tsym',Tsym,'B',B);
end


function M = compute_doppler_metrics(nu_norm, A_cut, params)
    % nu_norm : normalized Doppler (cycles/sample), typically [-0.5,0.5]
    % A_cut   : complex A(0,nu)
    % params  : struct like above (N, Fs, optional B, Tsym, Tobs)
    % Returns: 3 dB width in cycles/sample and Hz; PSLR; ISLR
    %
    % =========================================================================
    % Author    : Dr. (Eric) Hyeon Seok Rou
    % Version   : v1.0
    % Date      : Oct 13, 2025
    % =========================================================================
    
    N   = params.N;
    Fs  = params.Fs;
    Ts  = 1 / Fs;
    if isfield(params,'Tobs') && ~isempty(params.Tobs)
        Tobs = params.Tobs;
    else
        Tobs = N / Fs;
    end
    B = []; if isfield(params,'B'), B = params.B; end
    Tsym = []; if isfield(params,'Tsym'), Tsym = params.Tsym; end
    
    mag = abs(A_cut);
    mag = mag / max(mag);
    pwr = mag.^2;
    
    [~, ipk] = max(mag);
    [iL, iR] = first_minima_around_peak(mag, ipk);
    if isempty(iL) || isempty(iR) || iL>=iR
        [iL, iR] = halfpower_bounds(mag, ipk);
    end
    
    nL = halfpower_crossing(nu_norm, mag, ipk, -1);
    nR = halfpower_crossing(nu_norm, mag, ipk, +1);
    width_nu = nR - nL;          % cycles/sample (normalized Doppler)
    width_Hz = width_nu * Fs;    % Hz (since f_D = nu * Fs)
    
    max_sl = max([peak_outside(mag, 1, iL-1), peak_outside(mag, iR+1, numel(mag))]);
    if isempty(max_sl) || isnan(max_sl), max_sl = 0; end
    PSLR_dB = 20*log10(max_sl + eps);
    
    P_main = sum(pwr(iL:iR));
    P_side = sum(pwr) - P_main;
    ISLR_dB = 10*log10( max(P_side,eps) / max(P_main,eps) );
    
    M = struct();
    M.width_norm   = width_nu;           % cycles/sample
    M.width_Hz     = width_Hz;           % Hz
    M.width_perT   = width_Hz * Tobs;    % dimensionless ~ (width * Tobs); mainlobe ~ O(1)
    M.PSLR_dB      = PSLR_dB;
    M.ISLR_dB      = ISLR_dB;
    M.bounds_idx   = [iL iR];
    M.crossings    = [nL nR];
    M.scales       = struct('N',N,'Fs',Fs,'Ts',Ts,'Tobs',Tobs,'Tsym',Tsym,'B',B);
end


function [iL, iR] = halfpower_bounds(mag, ipk)
    hp = 1/sqrt(2);
    iL = find(mag(1:ipk) < hp, 1, 'last');
    if isempty(iL), iL = 1; end
    tmp = find(mag(ipk:end) < hp, 1, 'first');
    if isempty(tmp), iR = numel(mag);
    else, iR = ipk + tmp - 1;
    end
end


function xC = halfpower_crossing(x, mag, ipk, dirsign)
    hp = 1/sqrt(2);
    if dirsign < 0
        idx = ipk:-1:2;
        for k = 1:numel(idx)-1
            i1 = idx(k); 
            i0 = idx(k+1);
            if (mag(i0) >= hp && mag(i1) < hp) || (mag(i0) <= hp && mag(i1) > hp)
                t = (hp - mag(i0)) / (mag(i1) - mag(i0) + eps);
                xC = x(i0) + t*(x(i1) - x(i0));
                return;
            end
        end
        xC = x(1);
    else
        idx = ipk:1:numel(x)-1;
        for k = 1:numel(idx)-1
            i0 = idx(k); 
            i1 = idx(k+1);
            if (mag(i0) >= hp && mag(i1) < hp) || (mag(i0) <= hp && mag(i1) > hp)
                t = (hp - mag(i0)) / (mag(i1) - mag(i0) + eps);
                xC = x(i0) + t*(x(i1) - x(i0));
                return;
            end
        end
        xC = x(end);
    end
end

function mx = peak_outside(mag, a, b)
    if a>b, 
        mx = []; 
        return; 
    end
    mx = 0; have = false;
    for i = max(a,2):min(b-1,numel(mag)-1)
        if mag(i) >= mag(i-1) && mag(i) >= mag(i+1)
            if ~have || mag(i) > mx, 
                mx = mag(i); 
                have = true; 
            end
        end
    end
    if ~have, 
        mx = []; 
    end
end

function [iL, iR] = first_minima_around_peak(mag, ipk)
    iL = []; iR = [];
    prev = mag(ipk);
    for i = ipk-1:-1:2
        if mag(i) < mag(i-1) && mag(i) <= prev
            if mag(i) <= mag(i+1), 
                iL = i; 
                break; 
            end
        end
        prev = mag(i);
    end
    prev = mag(ipk);
    for i = ipk+1:1:numel(mag)-1
        if mag(i) < mag(i+1) && mag(i) <= prev
            if mag(i) <= mag(i-1), 
                iR = i; 
                break; 
            end
        end
        prev = mag(i);
    end
end







