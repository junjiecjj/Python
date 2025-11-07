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
