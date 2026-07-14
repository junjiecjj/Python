%% MAIN_STAGE3_OPTIMIZER_COMPARISON
% Stage 3.4 fair comparison for ZF-SNR-driven RIS phase optimizers.
%
% No Fig. 3/Fig. 4 reproduction is performed here.
%
% Methods compared on the same Hsr/Hrd in each trial:
%   1. random_single
%   2. random_best_of_numStarts
%   3. fixed_grid_zf_snr
%   4. coarse_to_fine_zf_snr
%   5. coarse_to_fine_zf_snr_with_condition_penalty

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

numTrials = 30;
numStarts = 3;
baseSeed = params0.repro.rngSeed + 4000;
methodNames = [
    "random_single"
    "random_best_of_numStarts"
    "fixed_grid_zf_snr"
    "coarse_to_fine_zf_snr"
    "coarse_to_fine_zf_snr_with_condition_penalty"
    ];
numMethods = numel(methodNames);

records = repmat(empty_record(), numTrials * numMethods, 1);
recordIdx = 0;

for trialIdx = 1:numTrials
    params = params0;
    params.repro.rngSeed = baseSeed + trialIdx;
    params.repro.resetRngInGenerateChannels = true;
    rng(baseSeed + 1000 + trialIdx, "twister");

    [Hsr, Hrd] = generate_channels(params);
    Nr = params.array.Nr_default;
    startPhases = exp(1j .* 2 .* pi .* rand(Nr, numStarts));

    randomSingleMetrics = evaluate_phase_metrics(Hsr, Hrd, startPhases(:, 1), params, "zf_snr");
    recordIdx = recordIdx + 1;
    records(recordIdx) = make_record(trialIdx, params.repro.rngSeed, "random_single", ...
        "none", "none", randomSingleMetrics, randomSingleMetrics, 0, 0, true, 0);

    [bestRandomMetrics, bestRandomStart] = best_random_phase(Hsr, Hrd, startPhases, params);
    recordIdx = recordIdx + 1;
    records(recordIdx) = make_record(trialIdx, params.repro.rngSeed, "random_best_of_numStarts", ...
        "none", "none", bestRandomMetrics, randomSingleMetrics, 0, bestRandomStart, true, 0);

    fixedOptions = common_options(startPhases, numStarts, baseSeed + 2000 + trialIdx);
    fixedOptions.searchMode = "fixed_grid";
    fixedOptions.phaseGridSize = 16;
    [vFixed, infoFixed, runtimeFixed] = run_optimizer(Hsr, Hrd, params, "zf_snr", fixedOptions);
    fixedMetrics = evaluate_phase_metrics(Hsr, Hrd, vFixed, params, "zf_snr");
    recordIdx = recordIdx + 1;
    records(recordIdx) = make_record(trialIdx, params.repro.rngSeed, "fixed_grid_zf_snr", ...
        "zf_snr", "fixed_grid", fixedMetrics, randomSingleMetrics, runtimeFixed, ...
        infoFixed.numStarts, infoFixed.converged, infoFixed.numIter);

    ctfOptions = common_options(startPhases, numStarts, baseSeed + 3000 + trialIdx);
    ctfOptions.searchMode = "coarse_to_fine";
    [vCtf, infoCtf, runtimeCtf] = run_optimizer(Hsr, Hrd, params, "zf_snr", ctfOptions);
    ctfMetrics = evaluate_phase_metrics(Hsr, Hrd, vCtf, params, "zf_snr");
    recordIdx = recordIdx + 1;
    records(recordIdx) = make_record(trialIdx, params.repro.rngSeed, "coarse_to_fine_zf_snr", ...
        "zf_snr", "coarse_to_fine", ctfMetrics, randomSingleMetrics, runtimeCtf, ...
        infoCtf.numStarts, infoCtf.converged, infoCtf.numIter);

    penaltyOptions = common_options(startPhases, numStarts, baseSeed + 4000 + trialIdx);
    penaltyOptions.searchMode = "coarse_to_fine";
    penaltyOptions.conditionPenaltyAlpha = 0.05;
    [vPenalty, infoPenalty, runtimePenalty] = run_optimizer( ...
        Hsr, Hrd, params, "zf_snr_with_condition_penalty", penaltyOptions);
    penaltyMetrics = evaluate_phase_metrics(Hsr, Hrd, vPenalty, params, "zf_snr");
    recordIdx = recordIdx + 1;
    records(recordIdx) = make_record(trialIdx, params.repro.rngSeed, ...
        "coarse_to_fine_zf_snr_with_condition_penalty", ...
        "zf_snr_with_condition_penalty", "coarse_to_fine", penaltyMetrics, ...
        randomSingleMetrics, runtimePenalty, infoPenalty.numStarts, ...
        infoPenalty.converged, infoPenalty.numIter);
end

records = records(1:recordIdx);
resultTable = struct2table(records);
summaryTable = build_summary_table(resultTable, methodNames);

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
logPath = fullfile(logDir, "stage3_optimizer_comparison_" + timestamp + ".txt");
dataPath = fullfile(dataDir, "stage3_optimizer_comparison_" + timestamp + ".mat");
pngPath = fullfile(figureDir, "stage3_optimizer_comparison.png");
figPath = fullfile(figureDir, "stage3_optimizer_comparison.fig");

save(dataPath, "resultTable", "summaryTable", "params0", "methodNames", ...
    "numTrials", "numStarts");
create_comparison_figure(resultTable, methodNames, pngPath, figPath);

logLines = [
    "Stage 3.4 optimizer comparison"
    "Trials: " + string(numTrials)
    "numStarts: " + string(numStarts)
    "Main objective: zf_snr"
    " "
    "Summary table:"
    formatted_table_lines(summaryTable)
    " "
    "Acceptance checks:"
    acceptance_lines(summaryTable)
    " "
    "Recommended main method: " + recommend_method(summaryTable)
    "Saved data: " + string(dataPath)
    "Saved PNG: " + string(pngPath)
    "Saved FIG: " + string(figPath)
    ];
writelines(logLines, logPath);
fprintf("%s\n", logLines);

function options = common_options(initialV, numStarts, rngSeed)
options = struct();
options.initialV = initialV;
options.numStarts = numStarts;
options.maxSweeps = 4;
options.phaseGridSize = 16;
options.coarseGridSize = 16;
options.fineGridSize = 8;
options.finerGridSize = 6;
options.fineHalfWidth = pi/12;
options.finerHalfWidth = pi/48;
options.tolerance = 1e-5;
options.rngSeed = rngSeed;
end

function [vOpt, info, runtimeSec] = run_optimizer(Hsr, Hrd, params, objectiveType, options)
timerStart = tic;
[vOpt, info] = optimize_ris_objective_driven(Hsr, Hrd, params, objectiveType, options);
runtimeSec = toc(timerStart);
end

function metrics = evaluate_phase_metrics(Hsr, Hrd, v, params, objectiveType)
[objectiveValue, evalMetrics] = evaluate_ris_objective(Hsr, Hrd, v, params, objectiveType, struct());
metrics = struct();
metrics.objectiveValue = objectiveValue;
metrics.snrDb = evalMetrics.snrDb;
metrics.snrLinear = evalMetrics.snrLinear;
metrics.pathGain = evalMetrics.pathGain;
metrics.condHeff = evalMetrics.condHeff;
metrics.zfRawPower = evalMetrics.zfRawPower;
metrics.unitModulusError = evalMetrics.unitModulusError;
end

function [bestMetrics, bestIdx] = best_random_phase(Hsr, Hrd, startPhases, params)
numStarts = size(startPhases, 2);
bestObjective = -Inf;
bestMetrics = struct();
bestIdx = 1;
for idx = 1:numStarts
    metrics = evaluate_phase_metrics(Hsr, Hrd, startPhases(:, idx), params, "zf_snr");
    if metrics.objectiveValue > bestObjective
        bestObjective = metrics.objectiveValue;
        bestMetrics = metrics;
        bestIdx = idx;
    end
end
end

function record = make_record(trialIdx, channelSeed, methodName, objectiveType, searchMode, ...
        metrics, baselineMetrics, runtimeSec, startCount, converged, numIter)
record = struct();
record.trialIdx = trialIdx;
record.channelSeed = string(channelSeed);
record.method = string(methodName);
record.objectiveType = string(objectiveType);
record.searchMode = string(searchMode);
record.snrDb = metrics.snrDb;
record.improvementVsRandomSingleDb = metrics.snrDb - baselineMetrics.snrDb;
record.pathGain = metrics.pathGain;
record.condHeff = metrics.condHeff;
record.zfRawPower = metrics.zfRawPower;
record.unitModulusError = metrics.unitModulusError;
record.runtimeSec = runtimeSec;
record.startCount = startCount;
record.converged = converged;
record.numIter = numIter;
end

function record = empty_record()
record = struct();
record.trialIdx = 0;
record.channelSeed = "";
record.method = "";
record.objectiveType = "";
record.searchMode = "";
record.snrDb = 0;
record.improvementVsRandomSingleDb = 0;
record.pathGain = 0;
record.condHeff = 0;
record.zfRawPower = 0;
record.unitModulusError = 0;
record.runtimeSec = 0;
record.startCount = 0;
record.converged = false;
record.numIter = 0;
end

function summaryTable = build_summary_table(resultTable, methodNames)
summary(numel(methodNames), 1) = struct();
randomBest = resultTable(resultTable.method == "random_best_of_numStarts", :);
for methodIdx = 1:numel(methodNames)
    methodName = methodNames(methodIdx);
    rows = resultTable(resultTable.method == methodName, :);
    improvementVsBestRandom = rows.snrDb - randomBest.snrDb;
    summary(methodIdx).method = methodName;
    summary(methodIdx).meanSnrDb = mean(rows.snrDb);
    summary(methodIdx).medianSnrDb = median(rows.snrDb);
    summary(methodIdx).stdSnrDb = std(rows.snrDb);
    summary(methodIdx).minSnrDb = min(rows.snrDb);
    summary(methodIdx).maxSnrDb = max(rows.snrDb);
    summary(methodIdx).meanImprovementVsRandomSingleDb = mean(rows.improvementVsRandomSingleDb);
    summary(methodIdx).meanImprovementVsRandomBestDb = mean(improvementVsBestRandom);
    summary(methodIdx).failureCountVsRandomSingle = nnz(rows.improvementVsRandomSingleDb <= 0);
    summary(methodIdx).failureCountVsRandomBest = nnz(improvementVsBestRandom <= 0);
    summary(methodIdx).meanCond = mean(rows.condHeff);
    summary(methodIdx).meanZfRawPower = mean(rows.zfRawPower);
    summary(methodIdx).meanPathGain = mean(rows.pathGain);
    summary(methodIdx).meanRuntimeSec = mean(rows.runtimeSec);
end
summaryTable = struct2table(summary);
end

function create_comparison_figure(resultTable, methodNames, pngPath, figPath)
fig = figure("Visible", "off", "Position", [100, 100, 1250, 800]);
tiledlayout(2, 3);

methodCat = categorical(resultTable.method, methodNames, methodNames);
methodCat = renamecats(methodCat, [
    "random single"
    "random best"
    "fixed grid"
    "coarse-to-fine"
    "ctf + cond penalty"
    ]);

nexttile;
boxchart(methodCat, resultTable.snrDb);
grid on; ylabel("SNR (dB)"); title("ZF-SNR");
xtickangle(25);

nexttile;
boxchart(methodCat, resultTable.improvementVsRandomSingleDb);
yline(0, "k-");
grid on; ylabel("Improvement (dB)"); title("Improvement vs random single");
xtickangle(25);

nexttile;
boxchart(methodCat, log10(resultTable.condHeff));
grid on; ylabel("log10(cond)"); title("Condition number");
xtickangle(25);

nexttile;
boxchart(methodCat, resultTable.runtimeSec);
grid on; ylabel("Runtime (s)"); title("Runtime");
xtickangle(25);

nexttile;
scatter(resultTable.condHeff, resultTable.snrDb, 24, methodCat, "filled");
set(gca, "XScale", "log");
grid on; xlabel("cond(Heff)"); ylabel("SNR (dB)");
title("SNR vs condition number");

nexttile;
boxchart(methodCat, log10(resultTable.zfRawPower));
grid on; ylabel("log10(ZF raw power)"); title("ZF raw power");
xtickangle(25);

saveas(fig, pngPath);
savefig(fig, figPath);
close(fig);
end

function lines = formatted_table_lines(tableInput) %#ok<INUSD>
text = evalc("disp(tableInput)");
lines = splitlines(string(text));
lines = lines(strlength(lines) > 0);
end

function lines = acceptance_lines(summaryTable)
ctf = summaryTable(summaryTable.method == "coarse_to_fine_zf_snr", :);
lines = [
    "coarse_to_fine_zf_snr mean improvement vs random_single > 0: " + string(ctf.meanImprovementVsRandomSingleDb > 0)
    "coarse_to_fine_zf_snr mean improvement vs random_best_of_numStarts > 0: " + string(ctf.meanImprovementVsRandomBestDb > 0)
    "coarse_to_fine_zf_snr failure count vs random_single: " + string(ctf.failureCountVsRandomSingle)
    "coarse_to_fine_zf_snr failure count vs random_best_of_numStarts: " + string(ctf.failureCountVsRandomBest)
    ];
end

function methodName = recommend_method(summaryTable)
candidateNames = [
    "fixed_grid_zf_snr"
    "coarse_to_fine_zf_snr"
    "coarse_to_fine_zf_snr_with_condition_penalty"
    ];
candidateRows = summaryTable(ismember(summaryTable.method, candidateNames), :);
[~, bestIdx] = max(candidateRows.meanSnrDb);
methodName = candidateRows.method(bestIdx);
end
