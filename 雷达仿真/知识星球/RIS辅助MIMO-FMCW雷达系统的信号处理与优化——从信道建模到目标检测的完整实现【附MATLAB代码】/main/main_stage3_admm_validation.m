%% MAIN_STAGE3_ADMM_VALIDATION
% Stage 3 diagnostic script for RIS phase optimization.
%
% This is no longer a narrow ADMM pass/fail script. It compares:
%   1. random phase;
%   2. quadratic ADMM approximation, which optimizes its quadratic proxy;
%   3. objective-driven optimizer with objectiveType = "path_gain";
%   4. objective-driven optimizer with objectiveType = "zf_snr".
%
% Fixed engineering model:
%   Hsr: Nr x Nt
%   Hrd: Nr x Nr
%   Heff = Hsr' * diag(v) * Hrd * diag(v)' * Hsr

clear; clc;

projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params();
rng(params.repro.rngSeed + 3, "twister");

logDir = fullfile(projectRoot, "outputs", "logs");
dataDir = fullfile(projectRoot, "outputs", "data");
figureDir = fullfile(projectRoot, "outputs", "figures");
if ~exist(logDir, "dir"), mkdir(logDir); end
if ~exist(dataDir, "dir"), mkdir(dataDir); end
if ~exist(figureDir, "dir"), mkdir(figureDir); end

[Hsr, Hrd, channelMeta] = generate_channels(params);
Nr = params.array.Nr_default;
Nt = params.array.Nt;
assert(isequal(size(Hsr), [Nr, Nt]), "Hsr must be Nr x Nt.");
assert(isequal(size(Hrd), [Nr, Nr]), "Hrd must be Nr x Nr.");

vRandom = exp(1j * 2 * pi * rand(Nr, 1));
randomResult = summarize_phase("random", "none", Hsr, Hrd, vRandom, params, "zf_snr", struct());

admmOptions = struct();
admmOptions.initialV = vRandom;
admmOptions.maxIter = 500;
admmOptions.tolerance = 1e-7;
admmOptions.rhoScale = 2;
[vAdmm, admmInfo] = optimize_ris_admm(Hsr, Hrd, params, admmOptions);
admmResult = summarize_phase("quadratic_admm", "quadratic_trace_proxy", ...
    Hsr, Hrd, vAdmm, params, "zf_snr", struct());

driverOptions = struct();
driverOptions.initialV = vRandom;
driverOptions.numStarts = 5;
driverOptions.maxSweeps = 12;
driverOptions.phaseGridSize = 24;
driverOptions.tolerance = 1e-5;
driverOptions.rngSeed = params.repro.rngSeed + 30;

[vPath, pathInfo] = optimize_ris_objective_driven(Hsr, Hrd, params, "path_gain", driverOptions);
pathResult = summarize_phase("objective_path_gain", "path_gain", ...
    Hsr, Hrd, vPath, params, "path_gain", struct());

[vSnr, snrInfo] = optimize_ris_objective_driven(Hsr, Hrd, params, "zf_snr", driverOptions);
snrResult = summarize_phase("objective_zf_snr", "zf_snr", ...
    Hsr, Hrd, vSnr, params, "zf_snr", struct());

results = [randomResult; admmResult; pathResult; snrResult];
resultTable = struct2table(results);

admmProxyImproved = admmInfo.quadraticObjectiveHistory(end) > admmInfo.quadraticObjectiveHistory(1);
pathOptimizerImproved = pathResult.pathGain > randomResult.pathGain;
snrOptimizerImproved = snrResult.snrDb > randomResult.snrDb + 0.1;

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
logPath = fullfile(logDir, "stage3_admm_validation_" + timestamp + ".txt");
dataPath = fullfile(dataDir, "stage3_admm_validation_" + timestamp + ".mat");
pngPath = fullfile(figureDir, "stage3_objective_diagnostics.png");
figPath = fullfile(figureDir, "stage3_objective_diagnostics.fig");

validation = struct();
validation.timestamp = timestamp;
validation.channelMeta = channelMeta;
validation.results = results;
validation.resultTable = resultTable;
validation.admmInfo = admmInfo;
validation.pathInfo = pathInfo;
validation.snrInfo = snrInfo;
validation.admmProxyImproved = admmProxyImproved;
validation.pathOptimizerImproved = pathOptimizerImproved;
validation.snrOptimizerImproved = snrOptimizerImproved;
validation.logPath = logPath;
validation.dataPath = dataPath;
validation.pngPath = pngPath;
validation.figPath = figPath;
save(dataPath, "validation", "params", "Hsr", "Hrd", ...
    "vRandom", "vAdmm", "vPath", "vSnr");

fig = figure("Visible", "off", "Position", [100, 100, 1100, 750]);
tiledlayout(2, 2);

nexttile;
plot(0:admmInfo.numIter, admmInfo.quadraticObjectiveHistory, "LineWidth", 1.4);
grid on; xlabel("Iteration"); ylabel("Quadratic proxy");
title("Quadratic ADMM proxy objective");

nexttile;
plot(0:admmInfo.numIter, admmInfo.truePathGainHistory, "LineWidth", 1.4);
hold on;
plot(1:pathInfo.numIter, pathInfo.bestPathGainHistory, "--", "LineWidth", 1.2);
plot(1:snrInfo.numIter, snrInfo.bestPathGainHistory, ":", "LineWidth", 1.2);
grid on; xlabel("Iteration"); ylabel("Path gain");
legend("ADMM proxy", "objective path_gain", "objective zf_snr", "Location", "best");
title("True path gain");

nexttile;
plot(0:admmInfo.numIter, admmInfo.zfSnrDbHistory, "LineWidth", 1.4);
hold on;
plot(1:pathInfo.numIter, pathInfo.bestSnrDbHistory, "--", "LineWidth", 1.2);
plot(1:snrInfo.numIter, snrInfo.bestSnrDbHistory, ":", "LineWidth", 1.2);
grid on; xlabel("Iteration"); ylabel("ZF-SNR (dB)");
legend("ADMM proxy", "objective path_gain", "objective zf_snr", "Location", "best");
title("ZF-normalized SNR");

nexttile;
semilogy(0:admmInfo.numIter, admmInfo.condHistory, "LineWidth", 1.4);
hold on;
semilogy(1:pathInfo.numIter, pathInfo.bestCondHistory, "--", "LineWidth", 1.2);
semilogy(1:snrInfo.numIter, snrInfo.bestCondHistory, ":", "LineWidth", 1.2);
grid on; xlabel("Iteration"); ylabel("cond(Heff)");
legend("ADMM proxy", "objective path_gain", "objective zf_snr", "Location", "best");
title("Condition number");

saveas(fig, pngPath);
savefig(fig, figPath);
close(fig);

logLines = build_log_lines(projectRoot, Hsr, Hrd, resultTable, admmInfo, ...
    pathInfo, snrInfo, admmProxyImproved, pathOptimizerImproved, ...
    snrOptimizerImproved, dataPath, pngPath, figPath);
writelines(logLines, logPath);

fprintf("%s\n", logLines);

function result = summarize_phase(name, optimizedObjective, Hsr, Hrd, v, params, objectiveType, options)
[objectiveValue, metrics] = evaluate_ris_objective(Hsr, Hrd, v, params, objectiveType, options);
result = struct();
result.method = string(name);
result.optimizedObjective = string(optimizedObjective);
result.reportedObjectiveType = string(objectiveType);
result.objectiveValue = objectiveValue;
result.pathGain = metrics.pathGain;
result.snrLinear = metrics.snrLinear;
result.snrDb = metrics.snrDb;
result.condHeff = metrics.condHeff;
result.zfRawPower = metrics.zfRawPower;
result.unitModulusError = metrics.unitModulusError;
end

function logLines = build_log_lines(projectRoot, Hsr, Hrd, resultTable, admmInfo, ...
    pathInfo, snrInfo, admmProxyImproved, pathOptimizerImproved, ...
    snrOptimizerImproved, dataPath, pngPath, figPath)

logLines = [
    "Stage 3 objective diagnostics"
    "Project root: " + string(projectRoot)
    "Hsr size: " + mat2str(size(Hsr))
    "Hrd size: " + mat2str(size(Hrd))
    " "
    "Result table:"
    formatted_table_lines(resultTable)
    " "
    "Diagnostic conclusions:"
    "Quadratic ADMM optimizes proxy objective: " + string(admmProxyImproved)
    "Path-gain objective optimizer improves true path_gain: " + string(pathOptimizerImproved)
    "ZF-SNR objective optimizer improves ZF-SNR by > 0.1 dB: " + string(snrOptimizerImproved)
    "ADMM method: " + string(admmInfo.method)
    "ADMM actual optimized objective: " + string(admmInfo.objectiveTypeActuallyOptimized)
    "ADMM rho: " + string(admmInfo.rho)
    "ADMM normT: " + string(admmInfo.normT)
    "ADMM final primal residual: " + string(admmInfo.primalResidualHistory(end))
    "ADMM final dual residual: " + string(admmInfo.dualResidualHistory(end))
    "ADMM converged: " + string(admmInfo.converged)
    "Path optimizer method: " + string(pathInfo.method)
    "Path optimizer final objective: " + string(pathInfo.finalObjective)
    "ZF-SNR optimizer method: " + string(snrInfo.method)
    "ZF-SNR optimizer final objective: " + string(snrInfo.finalObjective)
    " "
    "Interpretation:"
    "quadratic_admm optimizes a proxy, not the engineering SNR objective."
    "objective_path_gain optimizes true path_gain and may or may not optimize ZF-SNR."
    "objective_zf_snr directly optimizes the engineering target used for SNR curves."
    "Do not proceed to CD/Fig.3/Fig.4 unless the chosen optimizer and objective are accepted."
    " "
    "Saved data: " + string(dataPath)
    "Saved PNG: " + string(pngPath)
    "Saved FIG: " + string(figPath)
    ];
end

function lines = formatted_table_lines(tableInput) %#ok<INUSD>
text = evalc("disp(tableInput)");
lines = splitlines(string(text));
lines = lines(strlength(lines) > 0);
end
