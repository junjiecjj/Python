function [vSurrogate, info] = optimize_ris_surrogate(Hsr, Hrd, params, options)
%OPTIMIZE_RIS_SURROGATE Backup finite-difference phase surrogate optimizer.
%
%   Inputs:
%       Hsr     - Source/Radar-to-RIS channel, size Nr x Nt.
%       Hrd     - Stage-2 RIS-domain target/return effective channel, size
%                 Nr x Nr.
%       params  - Struct from config/paper_params.m.
%       options - Optional struct with initialV, maxIter, tolerance,
%                 gradientStep, minGradientStep, backtrackingFactor, and
%                 finiteDifferenceStep.
%
%   Outputs:
%       vSurrogate - RIS phase vector, size Nr x 1, with |v_i| = 1.
%       info       - Struct with objectiveHistory, numIter, converged, and
%                    method.
%
%   Notes:
%       This is not the paper ADMM. It is retained only as an explicit
%       comparison baseline for Stage-3 debugging.

arguments
    Hsr {mustBeNumeric}
    Hrd {mustBeNumeric}
    params struct
    options struct = struct()
end

Nr = size(Hsr, 1);
if isfield(options, "initialV")
    v = project_unit_modulus(options.initialV(:));
else
    v = exp(1j .* 2 .* pi .* rand(Nr, 1));
end

maxIter = get_option(options, "maxIter", 50);
tolerance = get_option(options, "tolerance", min(params.optim.tolerances));
gradientStep = get_option(options, "gradientStep", 0.005);
minGradientStep = get_option(options, "minGradientStep", 1e-6);
backtrackingFactor = get_option(options, "backtrackingFactor", 0.5);
finiteDifferenceStep = get_option(options, "finiteDifferenceStep", 1e-4);

objectiveHistory = zeros(maxIter + 1, 1);
stepHistory = zeros(maxIter, 1);
objectiveHistory(1) = compute_path_gain(Hsr, Hrd, v);
bestV = v;
bestObjective = objectiveHistory(1);
converged = false;

for iter = 1:maxIter
    previousObjective = objectiveHistory(iter);
    phaseGradient = finite_difference_phase_gradient(Hsr, Hrd, v, finiteDifferenceStep);
    phaseDirection = phaseGradient ./ max(norm(phaseGradient, inf), eps);
    theta = angle(v);
    step = gradientStep;
    accepted = false;

    while step >= minGradientStep
        candidateV = exp(1j .* (theta + step .* phaseDirection));
        candidateObjective = compute_path_gain(Hsr, Hrd, candidateV);
        if candidateObjective >= previousObjective
            accepted = true;
            break;
        end
        step = step .* backtrackingFactor;
    end

    if accepted
        v = candidateV;
        objectiveHistory(iter + 1) = candidateObjective;
    else
        objectiveHistory(iter + 1) = previousObjective;
        step = 0;
    end
    stepHistory(iter) = step;

    if objectiveHistory(iter + 1) > bestObjective
        bestObjective = objectiveHistory(iter + 1);
        bestV = v;
    end

    relativeChange = double(abs(objectiveHistory(iter + 1) - previousObjective) ...
        ./ max(abs(previousObjective), eps));
    if iter > 1 && all(relativeChange < tolerance)
        converged = true;
        break;
    end
end

vSurrogate = project_unit_modulus(bestV);
info = struct();
info.method = "finite_difference_phase_surrogate";
info.objectiveHistory = objectiveHistory(1:iter + 1);
info.stepHistory = stepHistory(1:iter);
info.numIter = iter;
info.converged = converged;
info.usesFiniteDifferenceGradient = true;
info.initialObjective = objectiveHistory(1);
info.finalObjective = compute_path_gain(Hsr, Hrd, vSurrogate);
info.unitModulusMaxError = max(abs(abs(vSurrogate) - 1));
end

function value = get_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = options.(fieldName);
else
    value = defaultValue;
end
end

function v = project_unit_modulus(z)
v = exp(1j .* angle(z));
zeroMask = abs(z) < eps;
v(zeroMask) = 1;
end

function grad = finite_difference_phase_gradient(Hsr, Hrd, v, delta)
Nr = numel(v);
theta = angle(v);
grad = zeros(Nr, 1);

for idx = 1:Nr
    thetaPlus = theta;
    thetaMinus = theta;
    thetaPlus(idx) = thetaPlus(idx) + delta;
    thetaMinus(idx) = thetaMinus(idx) - delta;

    gainPlus = compute_path_gain(Hsr, Hrd, exp(1j .* thetaPlus));
    gainMinus = compute_path_gain(Hsr, Hrd, exp(1j .* thetaMinus));
    grad(idx) = (gainPlus - gainMinus) ./ (2 .* delta);
end
end
