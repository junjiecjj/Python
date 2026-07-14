function result = main_stage4_pd_vs_snr(mode, options)
%MAIN_STAGE4_PD_VS_SNR Monte Carlo CA-CFAR Pd versus echo-SNR experiment.
%
%   Inputs:
%       mode    - "quick" or "full".
%                 quick uses a small Monte Carlo count for acceptance.
%                 full uses at least 100 trials unless overridden.
%       options - Optional struct:
%                 numTrials, echoSnrDb, saveOutputs, verbose,
%                 resampleChannelPerTrial, timestamp.
%
%   Outputs:
%       result - Struct with echo-SNR/noise axes, four-target Pd curves,
%                average Pd curves, hit counts, RIS gains, CFAR options,
%                saved file paths, and quick acceptance flags.
%
%   Units and model:
%       echoSnrDb is referenced to the optimized-RIS noiseless beat-signal
%       average sample power. The same echoNoisePower_W is used by No RIS,
%       Random RIS, and optimized RIS at each SNR point. No RIS remains the
%       blocked NLOS zero-target-echo baseline from Stage 4.
%
%   Detection rule:
%       Full-map CA-CFAR on the RD power map followed by truth-neighborhood
%       association through detect_rd_targets_cfar.m.

arguments
    mode (1,1) string {mustBeMember(mode, ["quick", "full"])} = "full"
    options struct = struct()
end

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

config = build_experiment_config(mode, options);
params = paper_params();
params.repro.rngSeed = params.repro.rngSeed + 6100;
params.repro.resetRngInGenerateChannels = true;

figureDir = fullfile(projectRoot, "outputs", "figures");
logDir = fullfile(projectRoot, "outputs", "logs");
dataDir = fullfile(projectRoot, "outputs", "data");
if config.saveOutputs
    ensure_directory(figureDir);
    ensure_directory(logDir);
    ensure_directory(dataDir);
end

targets = stage4_four_targets();
cfarOptions = stage4_cfar_options(params);
numTargets = numel(targets.range_m);
numSnr = numel(config.echoSnrDb);
methodNames = ["noRis", "randomRis", "optimizedRis"];
hitCounts = init_method_arrays(methodNames, numTargets, numSnr);
noisePowerByTrial_W = nan(config.numTrials, numSnr);
referenceSignalPowerByTrial_W = nan(config.numTrials, numSnr);
gainByTrial = init_method_gain_arrays(methodNames, config.numTrials, numSnr);
progressLines = strings(0, 1);
progressLines(end+1, 1) = "Stage 4 Pd-vs-SNR CA-CFAR Monte Carlo";
progressLines(end+1, 1) = "Mode: " + mode;
progressLines(end+1, 1) = "Trials: " + string(config.numTrials);
progressLines(end+1, 1) = "Echo SNR axis dB: " + strjoin(string(config.echoSnrDb), ", ");
progressLines(end+1, 1) = "No RIS model: blocked NLOS baseline with zero target echo amplitude";
progressLines(end+1, 1) = "SNR reference: optimized RIS noiseless beat-signal average sample power";

fixedBaselines = build_ris_baselines(params, targets, params.repro.rngSeed);
print_progress_header(config.verbose);
for snrIdx = 1:numSnr
    snrDb = config.echoSnrDb(snrIdx);
    for trialIdx = 1:config.numTrials
        if config.resampleChannelPerTrial
            trialSeed = params.repro.rngSeed + 1000 * snrIdx + trialIdx;
            baselines = build_ris_baselines(params, targets, trialSeed);
        else
            baselines = fixedBaselines;
        end

        referenceSignalPower_W = baselines.optimizedReferenceSignalPower_W;
        echoNoisePower_W = referenceSignalPower_W ./ 10.^(snrDb ./ 10);
        echoSeed = params.repro.rngSeed + 100000 + 1000 * snrIdx + trialIdx;
        detections = run_cfar_trial(params, targets, baselines, echoNoisePower_W, echoSeed, cfarOptions);

        hitCounts.noRis(:, snrIdx) = hitCounts.noRis(:, snrIdx) + detections.noRis.hit;
        hitCounts.randomRis(:, snrIdx) = hitCounts.randomRis(:, snrIdx) + detections.randomRis.hit;
        hitCounts.optimizedRis(:, snrIdx) = hitCounts.optimizedRis(:, snrIdx) + detections.optimizedRis.hit;
        noisePowerByTrial_W(trialIdx, snrIdx) = echoNoisePower_W;
        referenceSignalPowerByTrial_W(trialIdx, snrIdx) = referenceSignalPower_W;
        gainByTrial.noRis(trialIdx, snrIdx) = 0;
        gainByTrial.randomRis(trialIdx, snrIdx) = baselines.randomGain;
        gainByTrial.optimizedRis(trialIdx, snrIdx) = baselines.optimizedGain;

        currentPd = average_current_pd(hitCounts, snrIdx, trialIdx);
        line = sprintf("%-5s | SNR %6.1f dB | %3d/%3d | noise %.3e W | hits N/R/O %d/%d/%d | avgPd %.3f/%.3f/%.3f", ...
            char(mode), snrDb, trialIdx, config.numTrials, echoNoisePower_W, ...
            nnz(detections.noRis.hit), nnz(detections.randomRis.hit), nnz(detections.optimizedRis.hit), ...
            currentPd.noRis, currentPd.randomRis, currentPd.optimizedRis);
        progressLines(end+1, 1) = string(line); %#ok<AGROW>
        if config.verbose
            fprintf("%s\n", line);
        end
    end
end

pdPerTarget = divide_method_arrays(hitCounts, config.numTrials);
pdAverage = average_method_arrays(pdPerTarget);
echoNoisePower_W = mean(noisePowerByTrial_W, 1, "omitnan");
referenceSignalPower_W = mean(referenceSignalPowerByTrial_W, 1, "omitnan");
acceptance = assess_quick_acceptance(mode, pdAverage);

timestamp = config.timestamp;
figurePath = "";
dataPath = "";
logPath = "";
if config.saveOutputs
    figurePath = fullfile(figureDir, "stage4_pd_vs_snr_" + mode + "_" + timestamp + ".png");
    dataPath = fullfile(dataDir, "stage4_pd_vs_snr_" + mode + "_" + timestamp + ".mat");
    logPath = fullfile(logDir, "stage4_pd_vs_snr_" + mode + "_" + timestamp + ".txt");
    create_pd_vs_snr_figure(config.echoSnrDb, pdAverage, pdPerTarget, targets, figurePath);
end

result = struct();
result.mode = mode;
result.numTrials = config.numTrials;
result.echoSnrDb = config.echoSnrDb;
result.echoNoisePower_W = echoNoisePower_W;
result.noisePowerByTrial_W = noisePowerByTrial_W;
result.referenceSignalPower_W = referenceSignalPower_W;
result.referenceSignalPowerByTrial_W = referenceSignalPowerByTrial_W;
result.targets = targets;
result.cfarOptions = cfarOptions;
result.hitCounts = hitCounts;
result.pdPerTarget = pdPerTarget;
result.pdAverage = pdAverage;
result.gainByTrial = gainByTrial;
result.baselines = fixedBaselines;
result.config = config;
result.acceptance = acceptance;
result.figurePath = string(figurePath);
result.dataPath = string(dataPath);
result.logPath = string(logPath);
result.progressLines = progressLines;

summaryLines = build_summary_lines(result);
if config.saveOutputs
    save(dataPath, "result");
    writelines([progressLines; summaryLines], logPath);
end
if config.verbose
    fprintf("%s\n", summaryLines);
end
end

function config = build_experiment_config(mode, options)
config = struct();
config.mode = mode;
if mode == "quick"
    config.numTrials = 8;
    config.echoSnrDb = -40:5:-10;
    config.resampleChannelPerTrial = false;
else
    config.numTrials = 200;
    config.echoSnrDb = -40:2:0;
    config.resampleChannelPerTrial = false;
end
config.saveOutputs = true;
config.verbose = true;
config.timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
fieldNames = fieldnames(options);
for idx = 1:numel(fieldNames)
    config.(fieldNames{idx}) = options.(fieldNames{idx});
end
config.echoSnrDb = double(config.echoSnrDb(:).');
if ~(isscalar(config.numTrials) && isfinite(config.numTrials) && config.numTrials >= 1 && config.numTrials == round(config.numTrials))
    error("RIS_MIMO_FMCW:InvalidMonteCarloCount", "numTrials must be a positive integer.");
end
end

function targets = stage4_four_targets()
targets = struct();
targets.range_m = [25; 20; 10; 5];
targets.velocity_mps = [-1; 1; -1; 1];
targets.alpha = [1.00; 0.86; 0.74; 0.62];
end

function cfarOptions = stage4_cfar_options(params)
searchWindow = struct();
searchWindow.rangeHalfWidth_m = max(1.0, 3 * params.radar.rangeResolution);
searchWindow.velocityHalfWidth_mps = max(0.5, 3 * params.radar.velocityResolution);
cfarOptions = struct();
cfarOptions.trainingCells = [6, 6];
cfarOptions.guardCells = [2, 2];
cfarOptions.pfa = 1e-4;
cfarOptions.localMaxRadiusCells = [1, 1];
cfarOptions.associationRangeHalfWidth_m = searchWindow.rangeHalfWidth_m;
cfarOptions.associationVelocityHalfWidth_mps = searchWindow.velocityHalfWidth_mps;
end

function baselines = build_ris_baselines(params, targets, baselineSeed)
params.repro.rngSeed = baselineSeed;
params.repro.resetRngInGenerateChannels = true;
rng(baselineSeed, "twister");
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
fixedOptions.rngSeed = baselineSeed + 100;
[vOptimized, optInfo] = optimize_ris_objective_driven(Hsr, Hrd, params, "zf_snr", fixedOptions);
[randomGain, randomMetrics] = zf_effective_gain(Hsr, Hrd, vRandom, params);
[optimizedGain, optimizedMetrics] = zf_effective_gain(Hsr, Hrd, vOptimized, params);
[Yreference] = generate_fmcw_echo(params, targets, sqrt(optimizedGain), 0, struct("addNoise", false));

baselines = struct();
baselines.Hsr = Hsr;
baselines.Hrd = Hrd;
baselines.channelMeta = channelMeta;
baselines.vRandom = vRandom;
baselines.vOptimized = vOptimized;
baselines.optInfo = optInfo;
baselines.randomGain = randomGain;
baselines.optimizedGain = optimizedGain;
baselines.randomMetrics = randomMetrics;
baselines.optimizedMetrics = optimizedMetrics;
baselines.optimizedReferenceSignalPower_W = mean(abs(Yreference(:)).^2);
end

function detections = run_cfar_trial(params, targets, baselines, echoNoisePower_W, echoSeed, cfarOptions)
[YnoRis] = generate_fmcw_echo(params, targets, 0, echoNoisePower_W, struct("rngSeed", echoSeed));
[Yrandom] = generate_fmcw_echo(params, targets, sqrt(baselines.randomGain), echoNoisePower_W, struct("rngSeed", echoSeed));
[Yoptimized] = generate_fmcw_echo(params, targets, sqrt(baselines.optimizedGain), echoNoisePower_W, struct("rngSeed", echoSeed));
[RDnoRis, ~, rangeAxis, velocityAxis] = range_doppler_fft(YnoRis, params);
[RDrandom] = range_doppler_fft(Yrandom, params);
[RDoptimized] = range_doppler_fft(Yoptimized, params);
detections = struct();
detections.noRis = detect_rd_targets_cfar(RDnoRis, rangeAxis, velocityAxis, targets, cfarOptions);
detections.randomRis = detect_rd_targets_cfar(RDrandom, rangeAxis, velocityAxis, targets, cfarOptions);
detections.optimizedRis = detect_rd_targets_cfar(RDoptimized, rangeAxis, velocityAxis, targets, cfarOptions);
end

function arrays = init_method_arrays(methodNames, numTargets, numSnr)
arrays = struct();
for idx = 1:numel(methodNames)
    arrays.(methodNames(idx)) = zeros(numTargets, numSnr);
end
end

function arrays = init_method_gain_arrays(methodNames, numTrials, numSnr)
arrays = struct();
for idx = 1:numel(methodNames)
    arrays.(methodNames(idx)) = nan(numTrials, numSnr);
end
end

function arrays = divide_method_arrays(arrays, divisor)
methodNames = fieldnames(arrays);
for idx = 1:numel(methodNames)
    arrays.(methodNames{idx}) = arrays.(methodNames{idx}) ./ divisor;
end
end

function averages = average_method_arrays(arrays)
methodNames = fieldnames(arrays);
averages = struct();
for idx = 1:numel(methodNames)
    averages.(methodNames{idx}) = mean(arrays.(methodNames{idx}), 1);
end
end

function currentPd = average_current_pd(hitCounts, snrIdx, trialIdx)
currentPd = struct();
currentPd.noRis = mean(hitCounts.noRis(:, snrIdx)) ./ trialIdx;
currentPd.randomRis = mean(hitCounts.randomRis(:, snrIdx)) ./ trialIdx;
currentPd.optimizedRis = mean(hitCounts.optimizedRis(:, snrIdx)) ./ trialIdx;
end

function acceptance = assess_quick_acceptance(mode, pdAverage)
acceptance = struct();
acceptance.noRisLowest = mean(pdAverage.noRis) <= mean(pdAverage.randomRis) && ...
    mean(pdAverage.noRis) <= mean(pdAverage.optimizedRis);
acceptance.optimizedUsuallyBetter = mean(pdAverage.optimizedRis) >= mean(pdAverage.randomRis);
acceptance.optimizedRisesWithSnr = pdAverage.optimizedRis(end) >= pdAverage.optimizedRis(1);
acceptance.randomRisesWithSnr = pdAverage.randomRis(end) >= pdAverage.randomRis(1);
acceptance.overallRise = acceptance.optimizedRisesWithSnr && acceptance.randomRisesWithSnr;
acceptance.quickPassed = mode ~= "quick" || ...
    (acceptance.noRisLowest && acceptance.optimizedUsuallyBetter && acceptance.overallRise);
end

function create_pd_vs_snr_figure(echoSnrDb, pdAverage, pdPerTarget, targets, figurePath)
f = figure("Color", "w", "Position", [80, 80, 1280, 760]);
tiledlayout(f, 2, 2, "TileSpacing", "compact", "Padding", "compact");

nexttile([1, 2]);
plot(echoSnrDb, pdAverage.noRis, "-o", "LineWidth", 1.6, "DisplayName", "No RIS");
hold on;
plot(echoSnrDb, pdAverage.randomRis, "-s", "LineWidth", 1.6, "DisplayName", "Random RIS");
plot(echoSnrDb, pdAverage.optimizedRis, "-^", "LineWidth", 1.8, "DisplayName", "Fixed-grid optimized RIS");
grid on;
ylim([-0.05, 1.05]);
xlabel("Echo SNR (dB)");
ylabel("Average P_d");
title("Stage 4 CA-CFAR average detection probability");
legend("Location", "southeast");

nexttile;
plot_per_target_panel(echoSnrDb, pdPerTarget.randomRis, targets, "Random RIS per-target P_d");

nexttile;
plot_per_target_panel(echoSnrDb, pdPerTarget.optimizedRis, targets, "Optimized RIS per-target P_d");

exportgraphics(f, figurePath, "Resolution", 300);
close(f);
end

function plot_per_target_panel(echoSnrDb, pdMatrix, targets, panelTitle)
hold on;
for targetIdx = 1:size(pdMatrix, 1)
    plot(echoSnrDb, pdMatrix(targetIdx, :), "-o", "LineWidth", 1.2, ...
        "DisplayName", sprintf("T%d: %.0f m, %.0f m/s", targetIdx, targets.range_m(targetIdx), targets.velocity_mps(targetIdx)));
end
grid on;
ylim([-0.05, 1.05]);
xlabel("Echo SNR (dB)");
ylabel("P_d");
title(panelTitle);
legend("Location", "southeast");
end

function summaryLines = build_summary_lines(result)
summaryLines = [
    "Summary average Pd No RIS: " + strjoin(string(result.pdAverage.noRis), ", ")
    "Summary average Pd Random RIS: " + strjoin(string(result.pdAverage.randomRis), ", ")
    "Summary average Pd Optimized RIS: " + strjoin(string(result.pdAverage.optimizedRis), ", ")
    "Echo noise power W: " + strjoin(string(result.echoNoisePower_W), ", ")
    "Quick No RIS lowest: " + string(result.acceptance.noRisLowest)
    "Quick optimized >= random mean Pd: " + string(result.acceptance.optimizedUsuallyBetter)
    "Quick Pd rises overall: " + string(result.acceptance.overallRise)
    "Quick acceptance status: " + pass_fail(result.acceptance.quickPassed)
    "Saved figure PNG: " + result.figurePath
    "Saved data MAT: " + result.dataPath
    "Saved log TXT: " + result.logPath
    ];
end

function print_progress_header(verbose)
if verbose
    fprintf("Mode  | Echo SNR    | Trial   | Echo noise      | CFAR target hits | Running average Pd\n");
    fprintf("------+-------------+---------+-----------------+------------------+-----------------------------\n");
end
end

function [gain, metrics] = zf_effective_gain(Hsr, Hrd, v, params)
Heff = compute_effective_channel(Hsr, Hrd, v);
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

function ensure_directory(path)
if ~exist(path, "dir")
    mkdir(path);
end
end

function text = pass_fail(tf)
if tf
    text = "PASS";
else
    text = "FAIL";
end
end
