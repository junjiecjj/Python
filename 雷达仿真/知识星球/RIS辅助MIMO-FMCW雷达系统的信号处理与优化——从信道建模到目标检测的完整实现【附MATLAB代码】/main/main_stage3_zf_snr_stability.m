%% MAIN_STAGE3_ZF_SNR_STABILITY
% Stage 3.3 stability test for the ZF-SNR-driven RIS phase optimizer.
%
% Purpose:
%   Before reproducing Fig. 3/Fig. 4, verify that objectiveType = "zf_snr"
%   improves ZF-normalized SNR over random RIS phase across multiple channel
%   random seeds. Path gain is recorded only as an auxiliary metric.

clear; clc;

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params0 = paper_params();

logDir = fullfile(projectRoot, "outputs", "logs");
dataDir = fullfile(projectRoot, "outputs", "data");
figureDir = fullfile(projectRoot, "outputs", "figures");
if ~exist(logDir, "dir"), mkdir(logDir); end
if ~exist(dataDir, "dir"), mkdir(dataDir); end
if ~exist(figureDir, "dir"), mkdir(figureDir); end

numTrials = 8;
baseSeed = params0.repro.rngSeed + 300;

optimizerOptions = struct();
optimizerOptions.numStarts = 4;
optimizerOptions.maxSweeps = 10;
optimizerOptions.phaseGridSize = 24;
optimizerOptions.tolerance = 1e-5;

trial(numTrials, 1) = struct();

for trialIdx = 1:numTrials
    params = params0;
    params.repro.rngSeed = baseSeed + trialIdx;
    params.repro.resetRngInGenerateChannels = true;
    rng(baseSeed + 1000 + trialIdx, "twister");

    [Hsr, Hrd] = generate_channels(params);
    Nr = params.array.Nr_default;
    vRandom = exp(1j * 2 * pi * rand(Nr, 1));

    [~, randomMetrics] = evaluate_ris_objective(Hsr, Hrd, vRandom, params, "zf_snr", struct());

    optimizerOptions.initialV = vRandom;
    optimizerOptions.rngSeed = baseSeed + 2000 + trialIdx;
    [vOptimized, optInfo] = optimize_ris_objective_driven( ...
        Hsr, Hrd, params, "zf_snr", optimizerOptions);
    [~, optimizedMetrics] = evaluate_ris_objective(Hsr, Hrd, vOptimized, params, "zf_snr", struct());

    trial(trialIdx).trialIdx = trialIdx;
    trial(trialIdx).channelSeed = params.repro.rngSeed;
    trial(trialIdx).optimizerSeed = optimizerOptions.rngSeed;
    trial(trialIdx).randomSnrDb = randomMetrics.snrDb;
    trial(trialIdx).optimizedSnrDb = optimizedMetrics.snrDb;
    trial(trialIdx).snrImprovementDb = optimizedMetrics.snrDb - randomMetrics.snrDb;
    trial(trialIdx).randomPathGain = randomMetrics.pathGain;
    trial(trialIdx).optimizedPathGain = optimizedMetrics.pathGain;
    trial(trialIdx).randomCond = randomMetrics.condHeff;
    trial(trialIdx).optimizedCond = optimizedMetrics.condHeff;
    trial(trialIdx).randomZfRawPower = randomMetrics.zfRawPower;
    trial(trialIdx).optimizedZfRawPower = optimizedMetrics.zfRawPower;
    trial(trialIdx).unitModulusError = optimizedMetrics.unitModulusError;
    trial(trialIdx).numIter = optInfo.numIter;
    trial(trialIdx).converged = optInfo.converged;
    trial(trialIdx).bestHistoryEndpointMatches = abs(optInfo.bestSnrDbHistory(end) - optimizedMetrics.snrDb) < 1e-9;
end

trialTable = struct2table(trial);
failureMask = trialTable.snrImprovementDb <= 0;
failureCount = nnz(failureMask);
meanImprovementDb = mean(trialTable.snrImprovementDb);
medianImprovementDb = median(trialTable.snrImprovementDb);
minImprovementDb = min(trialTable.snrImprovementDb);
maxImprovementDb = max(trialTable.snrImprovementDb);

summary = struct();
summary.numTrials = numTrials;
summary.failureCount = failureCount;
summary.meanImprovementDb = meanImprovementDb;
summary.medianImprovementDb = medianImprovementDb;
summary.minImprovementDb = minImprovementDb;
summary.maxImprovementDb = maxImprovementDb;
summary.allHistoryEndpointsMatch = all(trialTable.bestHistoryEndpointMatches);

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
logPath = fullfile(logDir, "stage3_zf_snr_stability_" + timestamp + ".txt");
dataPath = fullfile(dataDir, "stage3_zf_snr_stability_" + timestamp + ".mat");
pngPath = fullfile(figureDir, "stage3_zf_snr_stability.png");
figPath = fullfile(figureDir, "stage3_zf_snr_stability.fig");

save(dataPath, "trial", "trialTable", "summary", "params0", "optimizerOptions");

fig = figure("Visible", "off", "Position", [100, 100, 1050, 650]);
tiledlayout(2, 2);

nexttile;
bar([trialTable.randomSnrDb, trialTable.optimizedSnrDb]);
grid on; xlabel("Trial"); ylabel("ZF-SNR (dB)");
legend("random", "optimized", "Location", "best");
title("Random vs optimized ZF-SNR");

nexttile;
bar(trialTable.snrImprovementDb);
yline(0, "k-");
grid on; xlabel("Trial"); ylabel("Improvement (dB)");
title("ZF-SNR improvement");

nexttile;
semilogy(trialTable.randomCond, "o-", "LineWidth", 1.1);
hold on;
semilogy(trialTable.optimizedCond, "s-", "LineWidth", 1.1);
grid on; xlabel("Trial"); ylabel("cond(Heff)");
legend("random", "optimized", "Location", "best");
title("Condition number");

nexttile;
semilogy(trialTable.randomZfRawPower, "o-", "LineWidth", 1.1);
hold on;
semilogy(trialTable.optimizedZfRawPower, "s-", "LineWidth", 1.1);
grid on; xlabel("Trial"); ylabel("ZF raw power");
legend("random", "optimized", "Location", "best");
title("ZF raw power");

saveas(fig, pngPath);
savefig(fig, figPath);
close(fig);

logLines = [
    "Stage 3.3 ZF-SNR stability test"
    "Objective optimized: zf_snr"
    "Trials: " + string(numTrials)
    "Failure count (improvement <= 0 dB): " + string(failureCount)
    "Mean improvement dB: " + string(meanImprovementDb)
    "Median improvement dB: " + string(medianImprovementDb)
    "Min improvement dB: " + string(minImprovementDb)
    "Max improvement dB: " + string(maxImprovementDb)
    "Best-so-far endpoints match final best: " + string(summary.allHistoryEndpointsMatch)
    " "
    "Trial table:"
    formatted_table_lines(trialTable)
    " "
    "Saved data: " + string(dataPath)
    "Saved PNG: " + string(pngPath)
    "Saved FIG: " + string(figPath)
    ];
writelines(logLines, logPath);

fprintf("%s\n", logLines);

function lines = formatted_table_lines(tableInput) %#ok<INUSD>
text = evalc("disp(tableInput)");
lines = splitlines(string(text));
lines = lines(strlength(lines) > 0);
end
