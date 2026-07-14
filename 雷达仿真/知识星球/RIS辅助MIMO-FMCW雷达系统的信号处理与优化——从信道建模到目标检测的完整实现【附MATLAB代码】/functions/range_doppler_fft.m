function [RD_complex, RD_dB, rangeAxis, velocityAxis, meta] = range_doppler_fft(Y, params, options)
%RANGE_DOPPLER_FFT Compute a range-Doppler spectrum from FMCW beat signals.
%
%   Inputs:
%       Y       - Complex beat signal matrix, size Nfast x Nchirp.
%                 Rows are fast-time samples, columns are slow-time chirps.
%       params  - Struct from config/paper_params.m.
%       options - Optional struct: nfftRange, nfftDoppler, useWindow.
%
%   Outputs:
%       RD_complex   - Complex RD spectrum after range FFT and Doppler FFT.
%                      Size nfftRange x nfftDoppler. Because Y is a complex
%                      dechirped signal, the full range-frequency interval is
%                      kept instead of a real-signal half spectrum.
%       RD_dB        - Magnitude map in dB, 20*log10(abs(RD_complex)).
%       rangeAxis    - Range axis in meters, size NrangeBins x 1.
%       velocityAxis - Velocity axis in m/s, size 1 x nfftDoppler.
%       meta         - Struct with FFT sizes and resolution values.

arguments
    Y {mustBeNumeric}
    params struct
    options struct = struct()
end

[Nfast, Nchirp] = size(Y);
nfftRange = get_option(options, "nfftRange", Nfast);
nfftDoppler = get_option(options, "nfftDoppler", Nchirp);
useWindow = get_option(options, "useWindow", true);

if useWindow
    rangeWindow = hann_local(Nfast);
    dopplerWindow = hann_local(Nchirp).';
    Yproc = Y .* rangeWindow .* dopplerWindow;
else
    Yproc = Y;
end

rangeSpectrum = fft(Yproc, nfftRange, 1);
numRangeBins = nfftRange;
RD_complex = fftshift(fft(rangeSpectrum, nfftDoppler, 2), 2);
RD_dB = 20 .* log10(abs(RD_complex) + eps);

fs = params.radar.sampleRate;
S = params.radar.slope;
c = params.radar.c;
lambda = params.radar.lambda;
Tc = params.radar.chirpTime;

rangeFreqAxis = (0:numRangeBins-1).' .* fs ./ nfftRange;
rangeAxis = rangeFreqAxis .* c ./ (2 .* S);
dopplerFreqAxis = (-floor(nfftDoppler/2):ceil(nfftDoppler/2)-1) ./ (nfftDoppler .* Tc);
velocityAxis = dopplerFreqAxis .* lambda ./ 2;

meta = struct();
meta.nfftRange = nfftRange;
meta.nfftDoppler = nfftDoppler;
meta.numRangeBins = numRangeBins;
meta.useWindow = useWindow;
meta.rangeResolution_m = c / (2 * params.radar.bandwidth);
meta.velocityResolution_mps = lambda / (2 * nfftDoppler * Tc);
meta.maxUnambiguousRange_m = fs * c / (2 * S);
meta.maxUnambiguousVelocity_mps = lambda / (4 * Tc);
end

function value = get_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = options.(fieldName);
else
    value = defaultValue;
end
end

function w = hann_local(N)
if N <= 1
    w = ones(N, 1);
else
    n = (0:N-1).';
    w = 0.5 - 0.5 .* cos(2 .* pi .* n ./ (N - 1));
end
end
