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
