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
%
% =========================================================================
% Author    : Dr. (Eric) Hyeon Seok Rou
% Version   : v1.0
% Date      : Oct 13, 2025
% =========================================================================

    % ----- defaults -----
    if OS_delay < 1, OS_delay = 1; end
    if Lh < 1, Lh = 1; end

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