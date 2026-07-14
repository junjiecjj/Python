function [vBest, info] = optimize_ris_objective_driven(Hsr, Hrd, params, objectiveType, options)
%OPTIMIZE_RIS_OBJECTIVE_DRIVEN Optimize RIS phases for an explicit objective.
%
%   Inputs:
%       Hsr           - Source/Radar-to-RIS channel, size Nr x Nt.
%       Hrd           - RIS-domain target/return effective channel, size Nr x Nr.
%       params        - Struct from config/paper_params.m.
%       objectiveType - Objective passed to evaluate_ris_objective:
%                       "path_gain", "zf_snr", or
%                       "zf_snr_with_condition_penalty".
%                       Default is "zf_snr".
%       options       - Optional struct:
%                       initialV, numStarts, maxSweeps, phaseGridSize,
%                       tolerance, conditionPenaltyAlpha, rngSeed,
%                       searchMode. initialV can be Nr x 1 or Nr x K.
%
%   Outputs:
%       vBest - Best unit-modulus RIS phase vector, size Nr x 1.
%       info  - Struct with per-record history and best-so-far history:
%               objectiveHistory, bestObjectiveHistory, pathGainHistory,
%               bestPathGainHistory, snrDbHistory, bestSnrDbHistory,
%               condHistory, bestCondHistory, stepHistory, startHistory,
%               sweepHistory, numIter, converged, method, and objectiveType.
%
%   Method:
%       Multi-start coordinate phase search. searchMode="fixed_grid" uses a
%       global fixed phase grid. searchMode="coarse_to_fine" first searches a
%       global coarse grid, then local grids around the best phase. This is not
%       ADMM. It is an engineering optimizer whose optimized objective is
%       exactly the requested objectiveType.

arguments
    Hsr {mustBeNumeric}
    Hrd {mustBeNumeric}
    params struct
    objectiveType {mustBeTextScalar} = "zf_snr"
    options struct = struct()
end

objectiveType = string(objectiveType);
Nr = size(Hsr, 1);
if size(Hrd, 1) ~= Nr || size(Hrd, 2) ~= Nr
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "Hrd must be Nr x Nr. Got Hrd %s and Nr=%d.", mat2str(size(Hrd)), Nr);
end

numStarts = get_option(options, "numStarts", 5);
maxSweeps = get_option(options, "maxSweeps", 12);
phaseGridSize = get_option(options, "phaseGridSize", 24);
searchMode = string(get_option(options, "searchMode", "fixed_grid"));
coarseGridSize = get_option(options, "coarseGridSize", phaseGridSize);
fineGridSize = get_option(options, "fineGridSize", 16);
finerGridSize = get_option(options, "finerGridSize", 12);
fineHalfWidth = get_option(options, "fineHalfWidth", pi/12);
finerHalfWidth = get_option(options, "finerHalfWidth", pi/48);
tolerance = get_option(options, "tolerance", 1e-5);
conditionPenaltyAlpha = get_option(options, "conditionPenaltyAlpha", 0.05);
objectiveOptions = struct("conditionPenaltyAlpha", conditionPenaltyAlpha);

if isfield(options, "rngSeed")
    rng(options.rngSeed, "twister");
end

initialCandidates = cell(numStarts, 1);
firstRandomStart = 1;
if isfield(options, "initialV")
    initialV = options.initialV;
    if size(initialV, 1) ~= Nr
        error("RIS_MIMO_FMCW:DimensionMismatch", ...
            "initialV must have Nr rows. Got initialV %s and Nr=%d.", mat2str(size(initialV)), Nr);
    end
    numProvidedStarts = min(size(initialV, 2), numStarts);
    for startIdx = 1:numProvidedStarts
        initialCandidates{startIdx} = project_unit_modulus(initialV(:, startIdx));
    end
    firstRandomStart = numProvidedStarts + 1;
end
for startIdx = firstRandomStart:numStarts
    initialCandidates{startIdx} = exp(1j .* 2 .* pi .* rand(Nr, 1));
end

maxRecords = numStarts * (maxSweeps + 1);
objectiveHistory = zeros(maxRecords, 1);
bestObjectiveHistory = zeros(maxRecords, 1);
pathGainHistory = zeros(maxRecords, 1);
bestPathGainHistory = zeros(maxRecords, 1);
snrDbHistory = zeros(maxRecords, 1);
bestSnrDbHistory = zeros(maxRecords, 1);
condHistory = zeros(maxRecords, 1);
bestCondHistory = zeros(maxRecords, 1);
zfRawPowerHistory = zeros(maxRecords, 1);
bestZfRawPowerHistory = zeros(maxRecords, 1);
stepHistory = zeros(maxRecords, 1);
startHistory = zeros(maxRecords, 1);
sweepHistory = zeros(maxRecords, 1);
recordIdx = 0;

bestObjective = -Inf;
bestV = initialCandidates{1};
bestMetrics = struct();
convergedAny = false;
phaseGrid = linspace(0, 2*pi, phaseGridSize + 1).';
phaseGrid(end) = [];
coarseGrid = linspace(0, 2*pi, coarseGridSize + 1).';
coarseGrid(end) = [];

for startIdx = 1:numStarts
    v = initialCandidates{startIdx};
    [currentObjective, currentMetrics] = evaluate_ris_objective( ...
        Hsr, Hrd, v, params, objectiveType, objectiveOptions);
    previousSweepObjective = currentObjective;
    [recordIdx, bestObjective, bestV, bestMetrics] = append_record( ...
        recordIdx, startIdx, 0, 0, currentObjective, currentMetrics, v);

    for sweepIdx = 1:maxSweeps
        acceptedUpdates = 0;
        for elementIdx = 1:Nr
            bestLocalObjective = currentObjective;
            bestLocalPhase = v(elementIdx);

            [bestLocalObjective, bestLocalPhase] = search_coordinate_phase( ...
                v, elementIdx, bestLocalObjective, bestLocalPhase);

            if bestLocalObjective > currentObjective
                v(elementIdx) = bestLocalPhase;
                currentObjective = bestLocalObjective;
                acceptedUpdates = acceptedUpdates + 1;
            end
        end

        [currentObjective, currentMetrics] = evaluate_ris_objective( ...
            Hsr, Hrd, v, params, objectiveType, objectiveOptions);

        [recordIdx, bestObjective, bestV, bestMetrics] = append_record( ...
            recordIdx, startIdx, sweepIdx, acceptedUpdates, currentObjective, currentMetrics, v);

        relativeImprovement = double(abs(currentObjective - previousSweepObjective) ...
            ./ max(abs(previousSweepObjective), eps));
        if all(relativeImprovement < tolerance) || acceptedUpdates == 0
            convergedAny = true;
            break;
        end
        previousSweepObjective = currentObjective;
    end
end

vBest = project_unit_modulus(bestV);
[finalObjective, finalMetrics] = evaluate_ris_objective( ...
    Hsr, Hrd, vBest, params, objectiveType, objectiveOptions);

info = struct();
info.method = "multi_start_coordinate_phase_search";
info.searchMode = searchMode;
info.objectiveType = objectiveType;
info.objectiveHistory = objectiveHistory(1:recordIdx);
info.bestObjectiveHistory = bestObjectiveHistory(1:recordIdx);
info.pathGainHistory = pathGainHistory(1:recordIdx);
info.bestPathGainHistory = bestPathGainHistory(1:recordIdx);
info.snrDbHistory = snrDbHistory(1:recordIdx);
info.bestSnrDbHistory = bestSnrDbHistory(1:recordIdx);
info.condHistory = condHistory(1:recordIdx);
info.bestCondHistory = bestCondHistory(1:recordIdx);
info.zfRawPowerHistory = zfRawPowerHistory(1:recordIdx);
info.bestZfRawPowerHistory = bestZfRawPowerHistory(1:recordIdx);
info.stepHistory = stepHistory(1:recordIdx);
info.startHistory = startHistory(1:recordIdx);
info.sweepHistory = sweepHistory(1:recordIdx);
info.numIter = recordIdx;
info.converged = convergedAny;
info.numStarts = numStarts;
info.maxSweeps = maxSweeps;
info.phaseGridSize = phaseGridSize;
info.coarseGridSize = coarseGridSize;
info.fineGridSize = fineGridSize;
info.finerGridSize = finerGridSize;
info.fineHalfWidth = fineHalfWidth;
info.finerHalfWidth = finerHalfWidth;
info.conditionPenaltyAlpha = conditionPenaltyAlpha;
info.initialObjective = info.objectiveHistory(1);
info.finalObjective = finalObjective;
info.finalMetrics = finalMetrics;
info.bestMetrics = bestMetrics;
info.unitModulusMaxError = max(abs(abs(vBest) - 1));

    function [newRecordIdx, newBestObjective, newBestV, newBestMetrics] = append_record( ...
            oldRecordIdx, startId, sweepId, acceptedUpdates, currentObjectiveValue, currentMetricsValue, currentV)
        newRecordIdx = oldRecordIdx + 1;
        objectiveHistory(newRecordIdx) = currentObjectiveValue;
        pathGainHistory(newRecordIdx) = currentMetricsValue.pathGain;
        snrDbHistory(newRecordIdx) = currentMetricsValue.snrDb;
        condHistory(newRecordIdx) = currentMetricsValue.condHeff;
        zfRawPowerHistory(newRecordIdx) = currentMetricsValue.zfRawPower;
        stepHistory(newRecordIdx) = acceptedUpdates;
        startHistory(newRecordIdx) = startId;
        sweepHistory(newRecordIdx) = sweepId;

        if currentObjectiveValue > bestObjective
            newBestObjective = currentObjectiveValue;
            newBestV = currentV;
            newBestMetrics = currentMetricsValue;
        else
            newBestObjective = bestObjective;
            newBestV = bestV;
            newBestMetrics = bestMetrics;
        end

        bestObjectiveHistory(newRecordIdx) = newBestObjective;
        bestPathGainHistory(newRecordIdx) = newBestMetrics.pathGain;
        bestSnrDbHistory(newRecordIdx) = newBestMetrics.snrDb;
        bestCondHistory(newRecordIdx) = newBestMetrics.condHeff;
        bestZfRawPowerHistory(newRecordIdx) = newBestMetrics.zfRawPower;
    end

    function [bestObjectiveOut, bestPhaseOut] = search_coordinate_phase( ...
            currentV, elementIdx, initialObjective, initialPhase)
        switch searchMode
            case "fixed_grid"
                [bestObjectiveOut, bestPhaseOut] = evaluate_phase_grid( ...
                    currentV, elementIdx, phaseGrid, initialObjective, initialPhase);
            case "coarse_to_fine"
                [coarseObjective, coarsePhase] = evaluate_phase_grid( ...
                    currentV, elementIdx, coarseGrid, initialObjective, initialPhase);
                fineCenter = angle(coarsePhase);
                fineGrid = local_phase_grid(fineCenter, fineHalfWidth, fineGridSize);
                [fineObjective, finePhase] = evaluate_phase_grid( ...
                    currentV, elementIdx, fineGrid, coarseObjective, coarsePhase);
                finerCenter = angle(finePhase);
                finerGrid = local_phase_grid(finerCenter, finerHalfWidth, finerGridSize);
                [bestObjectiveOut, bestPhaseOut] = evaluate_phase_grid( ...
                    currentV, elementIdx, finerGrid, fineObjective, finePhase);
            otherwise
                error("RIS_MIMO_FMCW:UnsupportedSearchMode", ...
                    "Unsupported searchMode: %s.", searchMode);
        end
    end

    function [bestObjectiveOut, bestPhaseOut] = evaluate_phase_grid( ...
            currentV, elementIdx, gridPhases, initialObjective, initialPhase)
        bestObjectiveOut = initialObjective;
        bestPhaseOut = initialPhase;
        for phaseIdx = 1:numel(gridPhases)
            candidateV = currentV;
            candidateV(elementIdx) = exp(1j .* gridPhases(phaseIdx));
            candidateObjective = evaluate_ris_objective( ...
                Hsr, Hrd, candidateV, params, objectiveType, objectiveOptions);

            if candidateObjective > bestObjectiveOut
                bestObjectiveOut = candidateObjective;
                bestPhaseOut = candidateV(elementIdx);
            end
        end
    end
end

function value = get_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = options.(fieldName);
else
    value = defaultValue;
end
end

function phases = local_phase_grid(centerPhase, halfWidth, numPoints)
if numPoints <= 1
    phases = centerPhase;
else
    phases = centerPhase + linspace(-halfWidth, halfWidth, numPoints).';
end
phases = mod(phases, 2*pi);
end

function v = project_unit_modulus(z)
v = exp(1j .* angle(z));
zeroMask = abs(z) < eps;
v(zeroMask) = 1;
end
