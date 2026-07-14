%% MAIN_STAGE2_MODEL_VALIDATION
% Stage 2 validation for the RIS-assisted MIMO-FMCW base model.
%
% Checks:
%   1. Parameters load from config/paper_params.m.
%   2. Channel dimensions are Hsr: Nr x Nt, Hrd: Nr x Nr.
%   3. Random RIS phase Phi is Nr x Nr with unit-modulus diagonal entries.
%   4. Equivalent channel Heff = Hsr' * Phi * Hrd * Phi' * Hsr is Nt x Nt.
%   5. ZF precoder satisfies ||B||_F^2 <= txPower_W.
%   6. SNR increases with transmit power.
%   7. SNR decreases with noise power.
%   8. Validation results are saved under outputs/logs and outputs/data.

clear; clc;

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params();
rng(params.repro.rngSeed, "twister");

logDir = fullfile(projectRoot, "outputs", "logs");
dataDir = fullfile(projectRoot, "outputs", "data");
if ~exist(logDir, "dir")
    mkdir(logDir);
end
if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
logPath = fullfile(logDir, "stage2_model_validation_" + timestamp + ".txt");
dataPath = fullfile(dataDir, "stage2_model_validation_" + timestamp + ".mat");

fprintf("Stage 2 model validation\n");
fprintf("Project root: %s\n", projectRoot);
fprintf("Random seed: %d\n", params.repro.rngSeed);

[Hsr, Hrd, channelMeta] = generate_channels(params);

Nt = params.array.Nt;
Nr = params.array.Nr_default;

assert(isequal(size(Hsr), [Nr, Nt]), "Hsr must be Nr x Nt.");
assert(isequal(size(Hrd), [Nr, Nr]), "Hrd must be Nr x Nr for the stage-2 equivalent model.");

v = exp(1j * 2 * pi * rand(Nr, 1));
Phi = params.optim.risReflectionAmplitude * diag(v);
assert(isequal(size(Phi), [Nr, Nr]), "Phi must be Nr x Nr.");
assert(max(abs(abs(diag(Phi)) - params.optim.risReflectionAmplitude)) < 1e-12, ...
    "Phi diagonal entries must have the configured reflection amplitude.");

Heff = Hsr' * Phi * Hrd * Phi' * Hsr;
assert(isequal(size(Heff), [Nt, Nt]), "Heff must be Nt x Nt.");

[B, zfInfo] = design_precoder_zf(Heff, params.power.txPower_W);
precoderPower = norm(B, "fro")^2;
assert(precoderPower <= params.power.txPower_W * (1 + 1e-10), ...
    "ZF precoder violates the transmit power constraint.");

[snrLinear, snrDb] = compute_snr(Heff, B, params.power.noisePower_W);
assert(isfinite(snrLinear) && snrLinear > 0, "SNR must be finite and positive.");
assert(isfinite(snrDb), "SNR dB must be finite.");

powerSweep_dBm = [0, 5, 10, 15, 20];
snrVsPower_dB = zeros(size(powerSweep_dBm));
for idx = 1:numel(powerSweep_dBm)
    txPower_W = params.utils.dbm2w(powerSweep_dBm(idx));
    [Btmp, ~] = design_precoder_zf(Heff, txPower_W);
    [~, snrVsPower_dB(idx)] = compute_snr(Heff, Btmp, params.power.noisePower_W);
end
assert(all(diff(snrVsPower_dB) > 0), "SNR must increase as transmit power increases.");

noiseSweep_dBm = [-20, -10, 0, 10, 20];
snrVsNoise_dB = zeros(size(noiseSweep_dBm));
for idx = 1:numel(noiseSweep_dBm)
    noisePower_W = params.utils.dbm2w(noiseSweep_dBm(idx));
    [~, snrVsNoise_dB(idx)] = compute_snr(Heff, B, noisePower_W);
end
assert(all(diff(snrVsNoise_dB) < 0), "SNR must decrease as noise power increases.");

validation = struct();
validation.timestamp = timestamp;
validation.dimensions.Hsr = size(Hsr);
validation.dimensions.Hrd = size(Hrd);
validation.dimensions.Phi = size(Phi);
validation.dimensions.Heff = size(Heff);
validation.dimensions.B = size(B);
validation.channelMeta = channelMeta;
validation.zfInfo = zfInfo;
validation.snrLinear = snrLinear;
validation.snrDb = snrDb;
validation.powerSweep_dBm = powerSweep_dBm;
validation.snrVsPower_dB = snrVsPower_dB;
validation.noiseSweep_dBm = noiseSweep_dBm;
validation.snrVsNoise_dB = snrVsNoise_dB;
validation.logPath = logPath;
validation.dataPath = dataPath;

save(dataPath, "validation", "params", "Hsr", "Hrd", "Phi", "Heff", "B");

logLines = [
    "Stage 2 model validation"
    "Project root: " + string(projectRoot)
    "Random seed: " + string(params.repro.rngSeed)
    "Hsr size: " + mat2str(size(Hsr))
    "Hrd size: " + mat2str(size(Hrd))
    "Phi size: " + mat2str(size(Phi))
    "Heff size: " + mat2str(size(Heff))
    "B size: " + mat2str(size(B))
    "ZF raw power W: " + string(zfInfo.rawPower_W)
    "ZF normalized power W: " + string(zfInfo.normalizedPower_W)
    "ZF relative error: " + string(zfInfo.relativeError)
    "Nominal SNR linear: " + string(snrLinear)
    "Nominal SNR dB: " + string(snrDb)
    "Power sweep dBm: " + mat2str(powerSweep_dBm)
    "SNR vs power dB: " + mat2str(snrVsPower_dB, 6)
    "Noise sweep dBm: " + mat2str(noiseSweep_dBm)
    "SNR vs noise dB: " + mat2str(snrVsNoise_dB, 6)
    "Saved data: " + string(dataPath)
    "Validation status: passed"
    ];
writelines(logLines, logPath);

fprintf("Dimensions:\n");
fprintf("  Hsr:  %s\n", mat2str(size(Hsr)));
fprintf("  Hrd:  %s\n", mat2str(size(Hrd)));
fprintf("  Phi:  %s\n", mat2str(size(Phi)));
fprintf("  Heff: %s\n", mat2str(size(Heff)));
fprintf("  B:    %s\n", mat2str(size(B)));
fprintf("ZF raw power: %.6e W\n", zfInfo.rawPower_W);
fprintf("ZF normalized power: %.6e W\n", zfInfo.normalizedPower_W);
fprintf("ZF relative error: %.6e\n", zfInfo.relativeError);
fprintf("Nominal SNR: %.6e linear, %.6f dB\n", snrLinear, snrDb);
fprintf("SNR vs transmit power dB: %s\n", mat2str(snrVsPower_dB, 6));
fprintf("SNR vs noise power dB: %s\n", mat2str(snrVsNoise_dB, 6));
fprintf("Saved log: %s\n", logPath);
fprintf("Saved data: %s\n", dataPath);
fprintf("Stage 2 model validation passed.\n");
