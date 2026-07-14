%% MAIN_STAGE4_RD_DETECTION
% Stage 4: four-target FMCW echo and range-Doppler detection validation.
%
% This script does not reproduce Fig. 3/Fig. 4 and does not use ADMM/CD.
% Main RIS optimizer: fixed_grid_zf_snr.

clear; clc;

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params();
params.repro.rngSeed = params.repro.rngSeed + 5000;
params.repro.resetRngInGenerateChannels = true;

figureDir = fullfile(projectRoot, "outputs", "figures");
logDir = fullfile(projectRoot, "outputs", "logs");
dataDir = fullfile(projectRoot, "outputs", "data");
if ~exist(figureDir, "dir"), mkdir(figureDir); end
if ~exist(logDir, "dir"), mkdir(logDir); end
if ~exist(dataDir, "dir"), mkdir(dataDir); end

rng(params.repro.rngSeed, "twister");
[Hsr, Hrd, channelMeta] = generate_channels(params);
Nr = size(Hsr, 1);

startPhases = exp(1j .* 2 .* pi .* rand(Nr, 3));
vRandom = startPhases(:, 1);

fixedOptions = struct();
fixedOptions.initialV = startPhases;
fixedOptions.numStarts = size(startPhases, 2);
fixedOptions.maxSweeps = 4;
fixedOptions.phaseGridSize = 16;
fixedOptions.searchMode = "fixed_grid";
fixedOptions.tolerance = 1e-5;
fixedOptions.rngSeed = params.repro.rngSeed + 100;
[vOptimized, optInfo] = optimize_ris_objective_driven( ...
    Hsr, Hrd, params, "zf_snr", fixedOptions);

[randomGain, randomMetrics] = zf_effective_gain(Hsr, Hrd, vRandom, params);
[optimizedGain, optimizedMetrics] = zf_effective_gain(Hsr, Hrd, vOptimized, params);
gainImprovementDb = 10 .* log10(optimizedGain ./ randomGain);

targets = struct();
targets.range_m = [25; 20; 10; 5];
targets.velocity_mps = [-1; 1; -1; 1];
targets.alpha = [1.00; 0.86; 0.74; 0.62];
numTargets = numel(targets.range_m);

% Controlled echo-domain noise for RD smoke validation. This is separate
% from the conservative link-budget noise used in Stage 2/3 SNR diagnostics.
echoNoisePower_W = 1e-12;
echoSeed = params.repro.rngSeed + 200;
[YnoRis, echoMetaNoRis] = generate_fmcw_echo( ...
    params, targets, 0, echoNoisePower_W, struct("rngSeed", echoSeed));
[Yrandom, echoMetaRandom] = generate_fmcw_echo( ...
    params, targets, sqrt(randomGain), echoNoisePower_W, struct("rngSeed", echoSeed));
[Yoptimized, echoMetaOptimized] = generate_fmcw_echo( ...
    params, targets, sqrt(optimizedGain), echoNoisePower_W, struct("rngSeed", echoSeed));

[RDnoRis, RDnoRisDb, rangeAxis, velocityAxis, rdMeta] = range_doppler_fft(YnoRis, params);
[RDrandom, RDrandomDb] = range_doppler_fft(Yrandom, params);
[RDoptimized, RDoptimizedDb] = range_doppler_fft(Yoptimized, params);

searchWindow = struct();
searchWindow.rangeHalfWidth_m = max(1.0, 3 * rdMeta.rangeResolution_m);
searchWindow.velocityHalfWidth_m = max(0.5, 3 * rdMeta.velocityResolution_mps);
noRisDetection = detect_target_peaks(RDnoRisDb, rangeAxis, velocityAxis, targets, searchWindow);
randomDetection = detect_target_peaks(RDrandomDb, rangeAxis, velocityAxis, targets, searchWindow);
optimizedDetection = detect_target_peaks(RDoptimizedDb, rangeAxis, velocityAxis, targets, searchWindow);
rdPeakImprovementVsNoRisDb = optimizedDetection.peakDb - noRisDetection.peakDb;
rdPeakImprovementDb = optimizedDetection.peakDb - randomDetection.peakDb;

cfarOptions = struct();
cfarOptions.trainingCells = [6, 6];
cfarOptions.guardCells = [2, 2];
cfarOptions.pfa = 1e-4;
cfarOptions.localMaxRadiusCells = [1, 1];
cfarOptions.associationRangeHalfWidth_m = searchWindow.rangeHalfWidth_m;
cfarOptions.associationVelocityHalfWidth_mps = searchWindow.velocityHalfWidth_m;
noRisCfarDetection = detect_rd_targets_cfar(RDnoRis, rangeAxis, velocityAxis, targets, cfarOptions);
randomCfarDetection = detect_rd_targets_cfar(RDrandom, rangeAxis, velocityAxis, targets, cfarOptions);
optimizedCfarDetection = detect_rd_targets_cfar(RDoptimized, rangeAxis, velocityAxis, targets, cfarOptions);
cfarPeakImprovementVsNoRisDb = optimizedCfarDetection.peakDb - noRisCfarDetection.peakDb;
cfarPeakImprovementDb = optimizedCfarDetection.peakDb - randomCfarDetection.peakDb;

rangePass = all(abs(optimizedDetection.peakRange_m - targets.range_m) <= searchWindow.rangeHalfWidth_m);
velocityPass = all(abs(optimizedDetection.peakVelocity_mps - targets.velocity_mps) <= searchWindow.velocityHalfWidth_m);
gainPass = optimizedGain > randomGain;
rdPeakPass = all(rdPeakImprovementDb > 1);
validationPassed = rangePass && velocityPass && gainPass && rdPeakPass;
cfarRangePass = all(optimizedCfarDetection.hit & ...
    abs(optimizedCfarDetection.peakRange_m - targets.range_m) <= searchWindow.rangeHalfWidth_m);
cfarVelocityPass = all(optimizedCfarDetection.hit & ...
    abs(optimizedCfarDetection.peakVelocity_mps - targets.velocity_mps) <= searchWindow.velocityHalfWidth_m);
cfarRandomHitPass = all(randomCfarDetection.hit);
cfarOptimizedHitPass = all(optimizedCfarDetection.hit);
cfarPeakPass = all(cfarPeakImprovementDb(randomCfarDetection.hit & optimizedCfarDetection.hit) > 1);
cfarValidationPassed = cfarRangePass && cfarVelocityPass && gainPass && ...
    cfarRandomHitPass && cfarOptimizedHitPass && cfarPeakPass;

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
pngPath = fullfile(figureDir, "stage4_rd_detection_" + timestamp + ".png");
figPath = fullfile(figureDir, "stage4_rd_detection_" + timestamp + ".fig");
logPath = fullfile(logDir, "stage4_rd_detection_" + timestamp + ".txt");
dataPath = fullfile(dataDir, "stage4_rd_detection_" + timestamp + ".mat");
cfarPngPath = fullfile(figureDir, "stage4_rd_detection_cfar_" + timestamp + ".png");
cfarFigPath = fullfile(figureDir, "stage4_rd_detection_cfar_" + timestamp + ".fig");
cfarDataPath = fullfile(dataDir, "stage4_rd_detection_cfar_" + timestamp + ".mat");

create_stage4_figure(RDnoRisDb, RDrandomDb, RDoptimizedDb, rangeAxis, velocityAxis, ...
    targets, noRisDetection, randomDetection, optimizedDetection, rdPeakImprovementDb, pngPath, figPath);
create_stage4_figure(RDnoRisDb, RDrandomDb, RDoptimizedDb, rangeAxis, velocityAxis, ...
    targets, noRisCfarDetection, randomCfarDetection, optimizedCfarDetection, ...
    cfarPeakImprovementDb, cfarPngPath, cfarFigPath, "CA-CFAR");

detectionTable = table((1:numTargets).', targets.range_m(:), targets.velocity_mps(:), targets.alpha(:), ...
    noRisDetection.peakDb(:), randomDetection.peakDb(:), optimizedDetection.peakDb(:), ...
    rdPeakImprovementVsNoRisDb(:), rdPeakImprovementDb(:), ...
    noRisDetection.peakRange_m(:), randomDetection.peakRange_m(:), optimizedDetection.peakRange_m(:), ...
    noRisDetection.peakVelocity_mps(:), randomDetection.peakVelocity_mps(:), optimizedDetection.peakVelocity_mps(:), ...
    optimizedDetection.rangeError_m(:), optimizedDetection.velocityError_mps(:), ...
    VariableNames=["targetIdx", "trueRange_m", "trueVelocity_mps", "alpha", ...
    "noRisPeakDb", "randomPeakDb", "optimizedPeakDb", ...
    "peakImprovementVsNoRisDb", "peakImprovementVsRandomDb", ...
    "noRisPeakRange_m", "randomPeakRange_m", "optimizedPeakRange_m", ...
    "noRisPeakVelocity_mps", "randomPeakVelocity_mps", "optimizedPeakVelocity_mps", ...
    "optimizedRangeError_m", "optimizedVelocityError_mps"]);

sourceDataPath = fullfile(dataDir, "stage4_rd_four_targets_latest.mat");
sourceCsvPath = fullfile(dataDir, "stage4_rd_four_targets_detection_latest.csv");
cfarSourceDataPath = fullfile(dataDir, "stage4_rd_four_targets_cfar_latest.mat");
cfarSourceCsvPath = fullfile(dataDir, "stage4_rd_four_targets_cfar_detection_latest.csv");

cfarDetectionTable = table((1:numTargets).', targets.range_m(:), targets.velocity_mps(:), targets.alpha(:), ...
    noRisCfarDetection.hit(:), randomCfarDetection.hit(:), optimizedCfarDetection.hit(:), ...
    noRisCfarDetection.peakDb(:), randomCfarDetection.peakDb(:), optimizedCfarDetection.peakDb(:), ...
    cfarPeakImprovementVsNoRisDb(:), cfarPeakImprovementDb(:), ...
    noRisCfarDetection.peakRange_m(:), randomCfarDetection.peakRange_m(:), optimizedCfarDetection.peakRange_m(:), ...
    noRisCfarDetection.peakVelocity_mps(:), randomCfarDetection.peakVelocity_mps(:), optimizedCfarDetection.peakVelocity_mps(:), ...
    optimizedCfarDetection.rangeError_m(:), optimizedCfarDetection.velocityError_mps(:), ...
    VariableNames=["targetIdx", "trueRange_m", "trueVelocity_mps", "alpha", ...
    "noRisCfarHit", "randomCfarHit", "optimizedCfarHit", ...
    "noRisCfarPeakDb", "randomCfarPeakDb", "optimizedCfarPeakDb", ...
    "cfarPeakImprovementVsNoRisDb", "cfarPeakImprovementVsRandomDb", ...
    "noRisCfarPeakRange_m", "randomCfarPeakRange_m", "optimizedCfarPeakRange_m", ...
    "noRisCfarPeakVelocity_mps", "randomCfarPeakVelocity_mps", "optimizedCfarPeakVelocity_mps", ...
    "optimizedCfarRangeError_m", "optimizedCfarVelocityError_mps"]);

save(dataPath, "params", "channelMeta", "Hsr", "Hrd", "vRandom", "vOptimized", ...
    "optInfo", "randomGain", "optimizedGain", "randomMetrics", "optimizedMetrics", ...
    "targets", "echoNoisePower_W", "YnoRis", "Yrandom", "Yoptimized", ...
    "RDnoRis", "RDrandom", "RDoptimized", "RDnoRisDb", "RDrandomDb", "RDoptimizedDb", ...
    "rangeAxis", "velocityAxis", "rdMeta", "echoMetaNoRis", "echoMetaRandom", ...
    "echoMetaOptimized", "noRisDetection", "randomDetection", "optimizedDetection", ...
    "rdPeakImprovementVsNoRisDb", "rdPeakImprovementDb", "detectionTable", "validationPassed", ...
    "cfarOptions", "noRisCfarDetection", "randomCfarDetection", "optimizedCfarDetection", ...
    "cfarPeakImprovementVsNoRisDb", "cfarPeakImprovementDb", "cfarDetectionTable", "cfarValidationPassed");
save(cfarDataPath, "params", "channelMeta", "Hsr", "Hrd", "vRandom", "vOptimized", ...
    "optInfo", "randomGain", "optimizedGain", "randomMetrics", "optimizedMetrics", ...
    "targets", "echoNoisePower_W", "RDnoRis", "RDrandom", "RDoptimized", ...
    "RDnoRisDb", "RDrandomDb", "RDoptimizedDb", "rangeAxis", "velocityAxis", "rdMeta", ...
    "cfarOptions", "noRisCfarDetection", "randomCfarDetection", "optimizedCfarDetection", ...
    "cfarPeakImprovementVsNoRisDb", "cfarPeakImprovementDb", "cfarDetectionTable", "cfarValidationPassed");
save(sourceDataPath, "RDnoRisDb", "RDrandomDb", "RDoptimizedDb", "rangeAxis", "velocityAxis", ...
    "targets", "noRisDetection", "randomDetection", "optimizedDetection", ...
    "rdPeakImprovementVsNoRisDb", "rdPeakImprovementDb", ...
    "detectionTable", "randomGain", "optimizedGain", "randomMetrics", "optimizedMetrics", ...
    "gainImprovementDb", "validationPassed");
writetable(detectionTable, sourceCsvPath);
save(cfarSourceDataPath, "RDnoRisDb", "RDrandomDb", "RDoptimizedDb", "rangeAxis", "velocityAxis", ...
    "targets", "noRisCfarDetection", "randomCfarDetection", "optimizedCfarDetection", ...
    "cfarPeakImprovementVsNoRisDb", "cfarPeakImprovementDb", "cfarDetectionTable", ...
    "randomGain", "optimizedGain", "randomMetrics", "optimizedMetrics", ...
    "gainImprovementDb", "cfarOptions", "cfarValidationPassed");
writetable(cfarDetectionTable, cfarSourceCsvPath);

logLines = [
    "Stage 4 FMCW RD detection validation"
    "Optimizer: fixed_grid_zf_snr"
    "No RIS model: blocked NLOS baseline with zero target echo amplitude and matched echo noise"
    "Targets: 4"
    "Target range m: " + strjoin(string(targets.range_m.'), ", ")
    "Target velocity m/s: " + strjoin(string(targets.velocity_mps.'), ", ")
    "Random G_ZF: " + string(randomGain)
    "Optimized G_ZF: " + string(optimizedGain)
    "G_ZF improvement dB: " + string(gainImprovementDb)
    "Random SNR dB: " + string(randomMetrics.snrDb)
    "Optimized SNR dB: " + string(optimizedMetrics.snrDb)
    "No RIS RD local noise peak dB: " + strjoin(string(noRisDetection.peakDb.'), ", ")
    "Random RD peak dB: " + strjoin(string(randomDetection.peakDb.'), ", ")
    "Random peak range m: " + strjoin(string(randomDetection.peakRange_m.'), ", ")
    "Random peak velocity m/s: " + strjoin(string(randomDetection.peakVelocity_mps.'), ", ")
    "Optimized RD peak dB: " + strjoin(string(optimizedDetection.peakDb.'), ", ")
    "Optimized peak range m: " + strjoin(string(optimizedDetection.peakRange_m.'), ", ")
    "Optimized peak velocity m/s: " + strjoin(string(optimizedDetection.peakVelocity_mps.'), ", ")
    "RD peak improvement vs no RIS dB: " + strjoin(string(rdPeakImprovementVsNoRisDb.'), ", ")
    "RD peak improvement dB: " + strjoin(string(rdPeakImprovementDb.'), ", ")
    "Range pass: " + string(rangePass)
    "Velocity pass: " + string(velocityPass)
    "Gain pass: " + string(gainPass)
    "RD peak pass (>1 dB): " + string(rdPeakPass)
    "Validation status: " + string(pass_fail(validationPassed))
    "Saved figure PNG: " + string(pngPath)
    "Saved figure FIG: " + string(figPath)
    "Saved data MAT: " + string(dataPath)
    "Saved Python source MAT: " + string(sourceDataPath)
    "Saved detection CSV: " + string(sourceCsvPath)
    "CA-CFAR training cells: " + mat2str(cfarOptions.trainingCells)
    "CA-CFAR guard cells: " + mat2str(cfarOptions.guardCells)
    "CA-CFAR pfa: " + string(cfarOptions.pfa)
    "No RIS CA-CFAR peak count / target hits: " + string(noRisCfarDetection.numCfarPeaks) + " / " + string(noRisCfarDetection.numAssociatedTargets)
    "Random CA-CFAR peak count / target hits: " + string(randomCfarDetection.numCfarPeaks) + " / " + string(randomCfarDetection.numAssociatedTargets)
    "Optimized CA-CFAR peak count / target hits: " + string(optimizedCfarDetection.numCfarPeaks) + " / " + string(optimizedCfarDetection.numAssociatedTargets)
    "Random CA-CFAR peak dB: " + strjoin(string(randomCfarDetection.peakDb.'), ", ")
    "Optimized CA-CFAR peak dB: " + strjoin(string(optimizedCfarDetection.peakDb.'), ", ")
    "CA-CFAR peak improvement dB: " + strjoin(string(cfarPeakImprovementDb.'), ", ")
    "CA-CFAR random hit pass: " + string(cfarRandomHitPass)
    "CA-CFAR optimized hit pass: " + string(cfarOptimizedHitPass)
    "CA-CFAR validation status: " + string(pass_fail(cfarValidationPassed))
    "Saved CFAR figure PNG: " + string(cfarPngPath)
    "Saved CFAR figure FIG: " + string(cfarFigPath)
    "Saved CFAR data MAT: " + string(cfarDataPath)
    "Saved CFAR Python source MAT: " + string(cfarSourceDataPath)
    "Saved CFAR detection CSV: " + string(cfarSourceCsvPath)
    ];
writelines(logLines, logPath);
fprintf("%s\n", logLines);

function [gain, metrics] = zf_effective_gain(Hsr, Hrd, v, params)
[Heff] = compute_effective_channel(Hsr, Hrd, v);
[B, zfInfo] = design_precoder_zf(Heff, params.power.txPower_W);
[snrLinear, snrDb] = compute_snr(Heff, B, params.power.noisePower_W);
gain = norm(Heff * B, "fro")^2;
metrics = struct();
metrics.Heff = Heff;
metrics.B = B;
metrics.zfInfo = zfInfo;
metrics.snrLinear = snrLinear;
metrics.snrDb = snrDb;
metrics.condHeff = cond(Heff);
metrics.zfRawPower = zfInfo.rawPower_W;
metrics.pathGain = norm(Heff, "fro")^2;
end

function detection = detect_target_peaks(RD_dB, rangeAxis, velocityAxis, targets, searchWindow)
numTargets = numel(targets.range_m);
detection = struct();
detection.peakDb = zeros(numTargets, 1);
detection.peakRange_m = zeros(numTargets, 1);
detection.peakVelocity_mps = zeros(numTargets, 1);
detection.rangeError_m = zeros(numTargets, 1);
detection.velocityError_mps = zeros(numTargets, 1);
detection.rangeIdx = zeros(numTargets, 1);
detection.velocityIdx = zeros(numTargets, 1);
for targetIdx = 1:numTargets
    rangeMask = abs(rangeAxis - targets.range_m(targetIdx)) <= searchWindow.rangeHalfWidth_m;
    velocityMask = abs(velocityAxis - targets.velocity_mps(targetIdx)) <= searchWindow.velocityHalfWidth_m;
    localMap = RD_dB(rangeMask, velocityMask);
    [peakDb, localLinearIdx] = max(localMap(:));
    [localRangeIdx, localVelocityIdx] = ind2sub(size(localMap), localLinearIdx);
    rangeIdxList = find(rangeMask);
    velocityIdxList = find(velocityMask);
    rangeIdx = rangeIdxList(localRangeIdx);
    velocityIdx = velocityIdxList(localVelocityIdx);
    detection.peakDb(targetIdx) = peakDb;
    detection.peakRange_m(targetIdx) = rangeAxis(rangeIdx);
    detection.peakVelocity_mps(targetIdx) = velocityAxis(velocityIdx);
    detection.rangeError_m(targetIdx) = detection.peakRange_m(targetIdx) - targets.range_m(targetIdx);
    detection.velocityError_mps(targetIdx) = detection.peakVelocity_mps(targetIdx) - targets.velocity_mps(targetIdx);
    detection.rangeIdx(targetIdx) = rangeIdx;
    detection.velocityIdx(targetIdx) = velocityIdx;
end
end

function create_stage4_figure(RDnoRisDb, RDrandomDb, RDoptimizedDb, rangeAxis, velocityAxis, ...
    targets, noRisDetection, randomDetection, optimizedDetection, rdPeakImprovementDb, pngPath, figPath, detectorName)
if nargin < 13
    detectorName = "Local peak";
end
fig = figure("Visible", "off", "Position", [100, 100, 1500, 720]);
tiledlayout(1, 4);

mapMax = max([RDnoRisDb(:); RDrandomDb(:); RDoptimizedDb(:)]);
mapLimits = [mapMax - 55, mapMax];

nexttile;
imagesc(velocityAxis, rangeAxis, RDnoRisDb);
axis xy; grid on; clim(mapLimits); colorbar;
xlabel("Velocity (m/s)"); ylabel("Range (m)");
title("No RIS RD map");
hold on;
plot(targets.velocity_mps, targets.range_m, "rx", "LineWidth", 1.5, "MarkerSize", 9);
plot(noRisDetection.peakVelocity_mps, noRisDetection.peakRange_m, "wo", "LineWidth", 1.2, "MarkerSize", 7);

nexttile;
imagesc(velocityAxis, rangeAxis, RDrandomDb);
axis xy; grid on; clim(mapLimits); colorbar;
xlabel("Velocity (m/s)"); ylabel("Range (m)");
title("Random RIS RD map");
hold on;
plot(targets.velocity_mps, targets.range_m, "rx", "LineWidth", 1.5, "MarkerSize", 9);
plot(randomDetection.peakVelocity_mps, randomDetection.peakRange_m, "wo", "LineWidth", 1.2, "MarkerSize", 7);

nexttile;
imagesc(velocityAxis, rangeAxis, RDoptimizedDb);
axis xy; grid on; clim(mapLimits); colorbar;
xlabel("Velocity (m/s)"); ylabel("Range (m)");
title("Fixed-grid ZF-SNR RIS RD map");
hold on;
plot(targets.velocity_mps, targets.range_m, "rx", "LineWidth", 1.5, "MarkerSize", 9);
plot(optimizedDetection.peakVelocity_mps, optimizedDetection.peakRange_m, "wo", "LineWidth", 1.2, "MarkerSize", 7);

nexttile;
targetLabels = categorical("T" + string(1:numel(targets.range_m)), ...
    "T" + string(1:numel(targets.range_m)), "Ordinal", true);
bar(targetLabels, [noRisDetection.peakDb(:), randomDetection.peakDb(:), optimizedDetection.peakDb(:)]);
grid on; ylabel("Local target peak (dB)");
title(detectorName + " target peak comparison (mean +" + compose("%.2f dB", mean(rdPeakImprovementDb, "omitnan")) + ")");
peakFloorDb = min([noRisDetection.peakDb(:); randomDetection.peakDb(:); optimizedDetection.peakDb(:)], [], "omitnan");
ylim([peakFloorDb - 5, 0]);
legend(["no RIS", "random", "optimized"], "Location", "southoutside", "Orientation", "horizontal");

sgtitle("Stage 4 four-target range-Doppler detection - " + detectorName);
saveas(fig, pngPath);
savefig(fig, figPath);
close(fig);
end

function textValue = pass_fail(flag)
if flag
    textValue = "PASS";
else
    textValue = "FAIL";
end
end
