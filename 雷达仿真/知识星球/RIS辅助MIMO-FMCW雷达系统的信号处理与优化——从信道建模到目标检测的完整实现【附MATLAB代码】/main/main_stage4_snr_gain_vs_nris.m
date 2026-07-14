function result = main_stage4_snr_gain_vs_nris(options)
%MAIN_STAGE4_SNR_GAIN_VS_NRIS Dense N_RIS sweep for ZF output gain.
%
%   Input:
%       options - Optional struct:
%                 NrisAxis, numTrials, phaseGridSize, numStarts,
%                 maxSweeps, tolerance, saveOutputs, verbose, timestamp.
%
%   Output:
%       result  - Struct containing dense N_RIS axis, raw Monte Carlo
%                 ZF-SNR/G_ZF/conditioning/runtime values, summary mean
%                 and standard-deviation curves, config, logs, and paths.
%
%   Matrix dimensions:
%       Each N_RIS point regenerates:
%           Hsr: Nr x Nt
%           Hrd: Nr x Nr
%           v:   Nr x 1
%       Random and optimized RIS are compared on the same channel.
%
%   Units:
%       SNR is recorded in dB. G_ZF = ||Heff * B||_F^2 uses linear power
%       units under the current normalized-ZF model. Runtime is seconds.

arguments
    options struct = struct()
end

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

config = build_config(options);
paramsBase = paper_params();
paramsBase.repro.rngSeed = paramsBase.repro.rngSeed + 8300;
paramsBase.repro.resetRngInGenerateChannels = true;

figureDir = fullfile(projectRoot, "outputs", "figures");
dataDir = fullfile(projectRoot, "outputs", "data");
logDir = fullfile(projectRoot, "outputs", "logs");
if config.saveOutputs
    ensure_directory(figureDir);
    ensure_directory(dataDir);
    ensure_directory(logDir);
end

numNr = numel(config.NrisAxis);
raw = init_raw_metrics(config.numTrials, numNr);
progressLines = [
    "Stage 4 ZF-SNR gain vs N_RIS"
    "N_RIS axis: " + strjoin(string(config.NrisAxis), ", ")
    "Trials: " + string(config.numTrials)
    "Optimizer: fixed-grid zf_snr"
    "phaseGridSize: " + string(config.phaseGridSize)
    "numStarts: " + string(config.numStarts)
    "maxSweeps: " + string(config.maxSweeps)
    ];
print_header(config.verbose);

for nrIdx = 1:numNr
    Nr = config.NrisAxis(nrIdx);
    params = set_nr(paramsBase, Nr);
    for trialIdx = 1:config.numTrials
        trialSeed = params.repro.rngSeed + 100000 * nrIdx + trialIdx;
        metrics = evaluate_nris_trial(params, trialSeed, config);
        raw = store_metrics(raw, nrIdx, trialIdx, metrics);
        current = current_summary(raw, nrIdx, trialIdx);
        line = sprintf("N_RIS %3d | trial %3d/%3d | random %8.3f dB | opt %8.3f dB | gain %7.3f dB | Ggain %7.3f dB | cond R/O %7.3f/%7.3f | runtime %.3f s", ...
            Nr, trialIdx, config.numTrials, metrics.randomSnrDb, metrics.optimizedSnrDb, ...
            metrics.snrGainDb, metrics.gzfGainDb, metrics.randomCond, metrics.optimizedCond, metrics.runtime_s);
        runningLine = sprintf("           running mean: random %8.3f dB | opt %8.3f dB | gain %7.3f dB | runtime %.3f s", ...
            current.randomSnrDbMean, current.optimizedSnrDbMean, current.snrGainDbMean, current.runtimeMean_s);
        progressLines(end+1, 1) = string(line); %#ok<AGROW>
        progressLines(end+1, 1) = string(runningLine); %#ok<AGROW>
        if config.verbose
            fprintf("%s\n", line);
            fprintf("%s\n", runningLine);
        end
    end
end

summary = summarize_raw(raw);
acceptance = struct();
acceptance.optimizedMeanAboveRandom = all(summary.snrGainDbMean > 0);
acceptance.outputsFinite = all(isfinite(summary.optimizedSnrDbMean)) && ...
    all(isfinite(summary.randomSnrDbMean)) && all(isfinite(summary.runtimeMean_s));

figurePath = "";
dataPath = "";
logPath = "";
if config.saveOutputs
    figurePath = fullfile(figureDir, "stage4_snr_gain_vs_nris_" + config.timestamp + ".png");
    dataPath = fullfile(dataDir, "stage4_snr_gain_vs_nris_" + config.timestamp + ".mat");
    logPath = fullfile(logDir, "stage4_snr_gain_vs_nris_" + config.timestamp + ".txt");
    create_gain_figure(config.NrisAxis, summary, figurePath);
end

result = struct();
result.NrisAxis = config.NrisAxis;
result.numTrials = config.numTrials;
result.config = config;
result.raw = raw;
result.summary = summary;
result.acceptance = acceptance;
result.figurePath = string(figurePath);
result.dataPath = string(dataPath);
result.logPath = string(logPath);
result.progressLines = progressLines;
result.summaryLines = build_summary_lines(result);
if config.saveOutputs
    save(dataPath, "result");
    writelines([progressLines; result.summaryLines], logPath);
end
if config.verbose
    fprintf("%s\n", result.summaryLines);
end
end

function config = build_config(options)
config = struct();
config.NrisAxis = 4:4:64;
config.numTrials = 100;
config.phaseGridSize = 16;
config.numStarts = 3;
config.maxSweeps = 4;
config.tolerance = 1e-5;
config.saveOutputs = true;
config.verbose = true;
config.timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
fields = fieldnames(options);
for fieldIdx = 1:numel(fields)
    config.(fields{fieldIdx}) = options.(fields{fieldIdx});
end
config.NrisAxis = double(config.NrisAxis(:).');
if isempty(config.NrisAxis) || any(config.NrisAxis < 1) || any(config.NrisAxis ~= round(config.NrisAxis))
    error("RIS_MIMO_FMCW:InvalidNrisAxis", "NrisAxis must contain positive integer sizes.");
end
if ~(isscalar(config.numTrials) && config.numTrials >= 1 && config.numTrials == round(config.numTrials))
    error("RIS_MIMO_FMCW:InvalidMonteCarloCount", "numTrials must be a positive integer.");
end
end

function params = set_nr(params, Nr)
params.array.Nr_default = Nr;
params.channel.Hsr_size = [Nr, params.array.Nt];
params.channel.Hrd_size = [Nr, Nr];
end

function metrics = evaluate_nris_trial(params, trialSeed, config)
params.repro.rngSeed = trialSeed;
params.repro.resetRngInGenerateChannels = true;
rng(trialSeed, "twister");
[Hsr, Hrd, channelMeta] = generate_channels(params);
Nr = size(Hsr, 1);
startPhases = exp(1j .* 2 .* pi .* rand(Nr, config.numStarts));
vRandom = startPhases(:, 1);

fixedOptions = struct();
fixedOptions.initialV = startPhases;
fixedOptions.numStarts = config.numStarts;
fixedOptions.maxSweeps = config.maxSweeps;
fixedOptions.phaseGridSize = config.phaseGridSize;
fixedOptions.searchMode = "fixed_grid";
fixedOptions.tolerance = config.tolerance;
fixedOptions.rngSeed = trialSeed + 100;
tic;
[vOptimized, optInfo] = optimize_ris_objective_driven(Hsr, Hrd, params, "zf_snr", fixedOptions);
runtime_s = toc;

[~, randomMetrics] = evaluate_ris_objective(Hsr, Hrd, vRandom, params, "zf_snr");
[~, optimizedMetrics] = evaluate_ris_objective(Hsr, Hrd, vOptimized, params, "zf_snr");
randomGzf = randomMetrics.snrLinear .* params.power.noisePower_W;
optimizedGzf = optimizedMetrics.snrLinear .* params.power.noisePower_W;

metrics = struct();
metrics.Nr = Nr;
metrics.channelMeta = channelMeta;
metrics.optInfo = optInfo;
metrics.randomSnrDb = randomMetrics.snrDb;
metrics.optimizedSnrDb = optimizedMetrics.snrDb;
metrics.snrGainDb = optimizedMetrics.snrDb - randomMetrics.snrDb;
metrics.randomGzf = randomGzf;
metrics.optimizedGzf = optimizedGzf;
metrics.gzfGainDb = 10 .* log10(optimizedGzf ./ randomGzf);
metrics.randomCond = randomMetrics.condHeff;
metrics.optimizedCond = optimizedMetrics.condHeff;
metrics.randomZfRawPower = randomMetrics.zfRawPower;
metrics.optimizedZfRawPower = optimizedMetrics.zfRawPower;
metrics.runtime_s = runtime_s;
end

function raw = init_raw_metrics(numTrials, numNr)
fields = ["randomSnrDb", "optimizedSnrDb", "snrGainDb", ...
    "randomGzf", "optimizedGzf", "gzfGainDb", ...
    "randomCond", "optimizedCond", "randomZfRawPower", ...
    "optimizedZfRawPower", "runtime_s"];
raw = struct();
for fieldIdx = 1:numel(fields)
    raw.(fields(fieldIdx)) = nan(numTrials, numNr);
end
end

function raw = store_metrics(raw, nrIdx, trialIdx, metrics)
fields = fieldnames(raw);
for fieldIdx = 1:numel(fields)
    raw.(fields{fieldIdx})(trialIdx, nrIdx) = metrics.(fields{fieldIdx});
end
end

function current = current_summary(raw, nrIdx, trialIdx)
current = struct();
current.randomSnrDbMean = mean(raw.randomSnrDb(1:trialIdx, nrIdx), "omitnan");
current.optimizedSnrDbMean = mean(raw.optimizedSnrDb(1:trialIdx, nrIdx), "omitnan");
current.snrGainDbMean = mean(raw.snrGainDb(1:trialIdx, nrIdx), "omitnan");
current.runtimeMean_s = mean(raw.runtime_s(1:trialIdx, nrIdx), "omitnan");
end

function summary = summarize_raw(raw)
fields = fieldnames(raw);
summary = struct();
for fieldIdx = 1:numel(fields)
    fieldName = fields{fieldIdx};
    summary.(fieldName + "Mean") = mean(raw.(fieldName), 1, "omitnan");
    summary.(fieldName + "Std") = std(raw.(fieldName), 0, 1, "omitnan");
end
summary.runtimeMean_s = summary.runtime_sMean;
summary.runtimeStd_s = summary.runtime_sStd;
end

function create_gain_figure(NrisAxis, summary, figurePath)
f = figure("Color", "w", "Position", [80, 80, 1380, 860]);
tiledlayout(f, 2, 2, "TileSpacing", "compact", "Padding", "compact");

nexttile;
plot_mean_std(NrisAxis, summary.randomSnrDbMean, summary.randomSnrDbStd, [0.10 0.45 0.75], "Random RIS");
hold on;
plot_mean_std(NrisAxis, summary.optimizedSnrDbMean, summary.optimizedSnrDbStd, [0.90 0.55 0.08], "Fixed-grid optimized RIS");
grid on;
xlabel("N_{RIS}");
ylabel("ZF output SNR (dB)");
title("ZF output SNR vs N_{RIS}");
legend("Location", "best");

nexttile;
plot_mean_std(NrisAxis, summary.snrGainDbMean, summary.snrGainDbStd, [0.18 0.55 0.25], "SNR gain");
grid on;
xlabel("N_{RIS}");
ylabel("Optimized - random SNR (dB)");
title("SNR gain vs N_{RIS}");

nexttile;
plot_mean_std(NrisAxis, summary.gzfGainDbMean, summary.gzfGainDbStd, [0.45 0.30 0.65], "G_{ZF} gain");
grid on;
xlabel("N_{RIS}");
ylabel("G_{ZF} gain (dB)");
title("G_{ZF} gain vs N_{RIS}");

nexttile;
plot_mean_std(NrisAxis, summary.runtimeMean_s, summary.runtimeStd_s, [0.78 0.22 0.20], "Runtime");
grid on;
xlabel("N_{RIS}");
ylabel("Optimizer runtime (s)");
title("Fixed-grid runtime vs N_{RIS}");

exportgraphics(f, figurePath, "Resolution", 300);
close(f);
end

function plot_mean_std(x, meanValues, stdValues, lineColor, displayName)
x = x(:).';
meanValues = meanValues(:).';
stdValues = stdValues(:).';
lower = meanValues - stdValues;
upper = meanValues + stdValues;
patch([x, fliplr(x)], [lower, fliplr(upper)], lineColor, ...
    "FaceAlpha", 0.14, "EdgeColor", "none", "HandleVisibility", "off");
hold on;
plot(x, meanValues, "-o", "Color", lineColor, "LineWidth", 1.8, ...
    "MarkerFaceColor", "w", "DisplayName", displayName);
end

function lines = build_summary_lines(result)
lines = [
    "Summary random SNR mean dB: " + join_values(result.summary.randomSnrDbMean)
    "Summary optimized SNR mean dB: " + join_values(result.summary.optimizedSnrDbMean)
    "Summary SNR gain mean dB: " + join_values(result.summary.snrGainDbMean)
    "Summary G_ZF gain mean dB: " + join_values(result.summary.gzfGainDbMean)
    "Summary runtime mean s: " + join_values(result.summary.runtimeMean_s)
    "Optimized mean SNR above random at all N_RIS: " + string(result.acceptance.optimizedMeanAboveRandom)
    "Summary finite: " + string(result.acceptance.outputsFinite)
    "Saved figure PNG: " + result.figurePath
    "Saved data MAT: " + result.dataPath
    "Saved log TXT: " + result.logPath
    ];
end

function text = join_values(values)
text = strjoin(string(values), ", ");
end

function print_header(verbose)
if verbose
    fprintf("N_RIS scan | Trial      | Random SNR | Optimized SNR | SNR gain | G_ZF gain | cond random/opt | runtime\n");
    fprintf("-----------+------------+------------+---------------+----------+-----------+-----------------+--------\n");
end
end

function ensure_directory(path)
if ~exist(path, "dir")
    mkdir(path);
end
end
