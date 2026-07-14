function [Y, meta] = generate_fmcw_echo(params, targets, amplitudeGain, noisePower_W, options)
%GENERATE_FMCW_ECHO Generate dechirped FMCW beat signals.
%
%   Inputs:
%       params         - Struct from config/paper_params.m.
%       targets        - Struct with fields:
%                         range_m, velocity_mps, alpha.
%                         Each field can be scalar or Q x 1.
%       amplitudeGain  - Scalar or Q x 1 amplitude gain from RIS/ZF link.
%                         Stage 4 uses sqrt(||Heff*B||_F^2).
%       noisePower_W   - Complex sample noise power in Watts.
%       options        - Optional struct with rngSeed and addNoise.
%
%   Outputs:
%       Y    - Complex beat signal matrix, size Nfast x Nchirp.
%              Rows are fast-time samples, columns are slow-time chirps.
%       meta - Struct with fb_Hz, fD_Hz, slope, lambda, Ts, Tc, and axes
%              resolution values.
%
%   Model:
%       y[n,m] = sum_q A_q exp(j*2*pi*(fb_q*n*Ts + fD_q*m*Tc)) + w[n,m]
%       fb_q = 2*S*R_q/c
%       fD_q = 2*v_q/lambda
%       S = B/Tchirp

arguments
    params struct
    targets struct
    amplitudeGain {mustBeNumeric} = 1
    noisePower_W (1,1) double {mustBeNonnegative} = 0
    options struct = struct()
end

range_m = targets.range_m(:);
velocity_mps = targets.velocity_mps(:);
alpha = targets.alpha(:);
numTargets = numel(range_m);
if numel(velocity_mps) ~= numTargets || numel(alpha) ~= numTargets
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "targets.range_m, targets.velocity_mps, and targets.alpha must have the same number of elements.");
end

amplitudeGain = amplitudeGain(:);
if isscalar(amplitudeGain)
    amplitudeGain = repmat(amplitudeGain, numTargets, 1);
elseif numel(amplitudeGain) ~= numTargets
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "amplitudeGain must be scalar or have one value per target.");
end

if isfield(options, "rngSeed")
    rng(options.rngSeed, "twister");
end
addNoise = get_option(options, "addNoise", true);

c = params.radar.c;
lambda = params.radar.lambda;
S = params.radar.slope;
Ts = 1 / params.radar.sampleRate;
Tc = params.radar.chirpTime;
Nfast = params.radar.numFastTimeSamples;
Nchirp = params.radar.numChirps;

n = (0:Nfast-1).';
m = 0:Nchirp-1;
Y = complex(zeros(Nfast, Nchirp));

fb_Hz = 2 .* S .* range_m ./ c;
fD_Hz = 2 .* velocity_mps ./ lambda;
complexAmplitude = amplitudeGain .* alpha;
for targetIdx = 1:numTargets
    fastPhase = exp(1j .* 2 .* pi .* fb_Hz(targetIdx) .* n .* Ts);
    slowPhase = exp(1j .* 2 .* pi .* fD_Hz(targetIdx) .* m .* Tc);
    Y = Y + complexAmplitude(targetIdx) .* (fastPhase * slowPhase);
end

if addNoise && noisePower_W > 0
    noise = sqrt(noisePower_W / 2) .* (randn(Nfast, Nchirp) + 1j .* randn(Nfast, Nchirp));
    Y = Y + noise;
end

meta = struct();
meta.model = "dechirped_fmcw_beat_signal";
meta.numTargets = numTargets;
meta.range_m = range_m;
meta.velocity_mps = velocity_mps;
meta.alpha = alpha;
meta.amplitudeGain = amplitudeGain;
meta.complexAmplitude = complexAmplitude;
meta.noisePower_W = noisePower_W;
meta.fb_Hz = fb_Hz;
meta.fD_Hz = fD_Hz;
meta.slope_HzPerSec = S;
meta.lambda_m = lambda;
meta.Ts_sec = Ts;
meta.Tc_sec = Tc;
meta.numFastTimeSamples = Nfast;
meta.numChirps = Nchirp;
meta.rangeResolution_m = c / (2 * params.radar.bandwidth);
meta.velocityResolution_mps = lambda / (2 * Nchirp * Tc);
end

function value = get_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = options.(fieldName);
else
    value = defaultValue;
end
end
