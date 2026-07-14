function [vAdmm, info] = optimize_ris_admm(Hsr, Hrd, params, options)
%OPTIMIZE_RIS_ADMM RIS phase optimization with closed-form ADMM updates.
%
%   Inputs:
%       Hsr     - Source/Radar-to-RIS channel, size Nr x Nt.
%       Hrd     - Stage-2 RIS-domain target/return effective channel, size
%                 Nr x Nr. The project convention is fixed as:
%                 Heff = Hsr' * diag(v) * Hrd * diag(v)' * Hsr.
%       params  - Struct from config/paper_params.m.
%       options - Optional struct:
%                 initialV  : Nr x 1 unit-modulus initial RIS phase.
%                 maxIter   : maximum ADMM iterations.
%                 tolerance : convergence tolerance.
%                 rho       : ADMM penalty. It is increased if needed so
%                             rho*I + T is numerically positive definite.
%
%   Outputs:
%       vAdmm - RIS phase vector, size Nr x 1, with |v_i| = 1.
%       info  - Struct containing quadraticObjectiveHistory,
%               truePathGainHistory, zfSnrHistory, condHistory,
%               primalResidualHistory, dualResidualHistory, numIter,
%               converged, rho, normT, method, T, and model notes.
%
%   ADMM form:
%       The paper uses an extended vector x in C^(Nr+1), unit-modulus u,
%       and multiplier mu:
%
%           u = exp(1j * angle(x - mu/rho))
%           x = (rho*I + T)^(-1) * (rho*u + mu)
%           mu = mu + rho*(u - x)
%
%       Under the current Hrd: Nr x Nr project convention, the executable
%       path gain ||Heff||_F^2 is quartic in v and cannot be represented by
%       the paper's quadratic T matrix without changing the model. This
%       function therefore constructs a dimension-consistent quadratic ADMM
%       approximation from the Hermitian part of trace(Heff):
%
%           trace(Heff) = v' * Q * v, Q = (Hsr*Hsr') .* transpose(Hrd)
%
%       The ADMM minimizes 0.5*x'*T*x with:
%
%           T(1:Nr,1:Nr) = -0.5*(Q + Q')
%           T(Nr+1,Nr+1) = 0
%
%       The final RIS phase is recovered as v = exp(1j*angle(x(1:Nr)/x(end))).
%       This is explicitly a quadratic ADMM approximation, not an exact
%       reproduction of the paper's T matrix.

arguments
    Hsr {mustBeNumeric}
    Hrd {mustBeNumeric}
    params struct
    options struct = struct()
end

Nr = size(Hsr, 1);
if size(Hrd, 1) ~= Nr || size(Hrd, 2) ~= Nr
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "Hrd must be Nr x Nr. Got Hrd %s and Nr=%d.", mat2str(size(Hrd)), Nr);
end

maxIter = get_option(options, "maxIter", params.optim.maxIter);
tolerance = get_option(options, "tolerance", min(params.optim.tolerances));
rhoRequested = get_option(options, "rho", []);
rhoScale = get_option(options, "rhoScale", 2);
conditionPenaltyAlpha = get_option(options, "conditionPenaltyAlpha", 0.05);

if isfield(options, "initialV")
    v0 = project_unit_modulus(options.initialV(:));
else
    v0 = exp(1j .* 2 .* pi .* rand(Nr, 1));
end

[T, qMatrix, tMeta] = build_quadratic_t_matrix(Hsr, Hrd);
minEigT = min(real(eig(T)));
rhoMinimum = max(0, -minEigT) + 1e-9;
normT = norm(T, 2);
if isempty(rhoRequested)
    rho = max(1.05 * rhoMinimum, rhoScale * max(normT, eps));
else
    rho = max(rhoRequested, 1.05 * rhoMinimum);
end

x = [v0; 1];
mu = zeros(Nr + 1, 1);
systemMatrix = rho .* eye(Nr + 1) + T;

quadraticObjectiveHistory = zeros(maxIter + 1, 1);
truePathGainHistory = zeros(maxIter + 1, 1);
zfSnrHistory = zeros(maxIter + 1, 1);
zfSnrDbHistory = zeros(maxIter + 1, 1);
condHistory = zeros(maxIter + 1, 1);
primalResidualHistory = zeros(maxIter, 1);
dualResidualHistory = zeros(maxIter, 1);

vCurrent = recover_phase_from_extended_x(x, Nr);
[~, initialMetrics] = evaluate_ris_objective(Hsr, Hrd, vCurrent, params, "zf_snr", ...
    struct("conditionPenaltyAlpha", conditionPenaltyAlpha));
truePathGainHistory(1) = initialMetrics.pathGain;
zfSnrHistory(1) = initialMetrics.snrLinear;
zfSnrDbHistory(1) = initialMetrics.snrDb;
condHistory(1) = initialMetrics.condHeff;
quadraticObjectiveHistory(1) = real(-(x' * T * x));

bestV = vCurrent;
bestQuadraticObjective = quadraticObjectiveHistory(1);
converged = false;

for iter = 1:maxIter
    xPrevious = x;

    u = project_unit_modulus(x - (mu ./ rho));
    x = systemMatrix \ (rho .* u + mu);
    mu = mu + rho .* (u - x);

    vCurrent = recover_phase_from_extended_x(x, Nr);
    [~, currentMetrics] = evaluate_ris_objective(Hsr, Hrd, vCurrent, params, "zf_snr", ...
        struct("conditionPenaltyAlpha", conditionPenaltyAlpha));
    quadraticObjective = real(-(x' * T * x));

    quadraticObjectiveHistory(iter + 1) = quadraticObjective;
    truePathGainHistory(iter + 1) = currentMetrics.pathGain;
    zfSnrHistory(iter + 1) = currentMetrics.snrLinear;
    zfSnrDbHistory(iter + 1) = currentMetrics.snrDb;
    condHistory(iter + 1) = currentMetrics.condHeff;
    primalResidualHistory(iter) = norm(u - x);
    dualResidualHistory(iter) = rho .* norm(x - xPrevious);

    if quadraticObjective >= bestQuadraticObjective
        bestQuadraticObjective = quadraticObjective;
        bestV = vCurrent;
    end

    primalOk = primalResidualHistory(iter) < sqrt(Nr + 1) .* tolerance;
    dualOk = dualResidualHistory(iter) < sqrt(Nr + 1) .* tolerance;
    if iter > 1 && primalOk && dualOk
        converged = true;
        break;
    end
end

numIter = iter;
vFinal = recover_phase_from_extended_x(x, Nr);

% Return the best iterate for ADMM's own quadratic proxy, not for path gain
% or SNR. This keeps the method's optimized objective explicit.
if bestQuadraticObjective >= quadraticObjectiveHistory(1)
    vAdmm = project_unit_modulus(bestV);
else
    vAdmm = project_unit_modulus(vFinal);
end

[~, finalMetrics] = evaluate_ris_objective(Hsr, Hrd, vAdmm, params, "zf_snr", ...
    struct("conditionPenaltyAlpha", conditionPenaltyAlpha));

info = struct();
info.method = "quadratic_admm_approximation";
info.paperLikeUpdates = true;
info.usesFiniteDifferenceGradient = false;
info.objectiveTypeActuallyOptimized = "quadratic_trace_proxy";
info.objective = "ADMM optimizes normalized real(trace(Heff)) quadratic proxy, not path_gain or zf_snr";
info.objectiveHistory = quadraticObjectiveHistory(1:numIter + 1);
info.quadraticObjectiveHistory = quadraticObjectiveHistory(1:numIter + 1);
info.truePathGainHistory = truePathGainHistory(1:numIter + 1);
info.zfSnrHistory = zfSnrHistory(1:numIter + 1);
info.zfSnrDbHistory = zfSnrDbHistory(1:numIter + 1);
info.condHistory = condHistory(1:numIter + 1);
info.primalResidualHistory = primalResidualHistory(1:numIter);
info.dualResidualHistory = dualResidualHistory(1:numIter);
info.numIter = numIter;
info.converged = converged;
info.rho = rho;
info.rhoRequested = rhoRequested;
info.rhoScale = rhoScale;
info.rhoMinimum = rhoMinimum;
info.normT = normT;
info.tolerance = tolerance;
info.initialObjective = quadraticObjectiveHistory(1);
info.finalObjective = info.objectiveHistory(end);
info.bestObjective = bestQuadraticObjective;
info.finalPathGain = finalMetrics.pathGain;
info.finalZfSnrLinear = finalMetrics.snrLinear;
info.finalZfSnrDb = finalMetrics.snrDb;
info.finalCondHeff = finalMetrics.condHeff;
info.unitModulusMaxError = max(abs(abs(vAdmm) - 1));
info.T = T;
info.Q = qMatrix;
info.TMeta = tMeta;
info.dimension.x = [Nr + 1, 1];
info.dimension.u = [Nr + 1, 1];
info.dimension.mu = [Nr + 1, 1];
info.dimension.T = size(T);
end

function [T, Qh, meta] = build_quadratic_t_matrix(Hsr, Hrd)
Nr = size(Hsr, 1);
Rsr = Hsr * Hsr';
Q = Rsr .* transpose(Hrd);
QhUnscaled = 0.5 .* (Q + Q');
scale = max(norm(QhUnscaled, "fro"), eps);
Qh = QhUnscaled ./ scale;

T = zeros(Nr + 1, Nr + 1);
T(1:Nr, 1:Nr) = -Qh;
T = 0.5 .* (T + T');

meta = struct();
meta.proxy = "real(trace(Heff)) = real(v^H * Qh * v)";
meta.QSize = size(Qh);
meta.TSize = size(T);
meta.QFroNormBeforeScaling = scale;
meta.TwoNormAfterScaling = norm(T, 2);
meta.note = "Quadratic proxy is dimension-consistent but not equal to ||Heff||_F^2.";
end

function value = get_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = options.(fieldName);
else
    value = defaultValue;
end
end

function v = recover_phase_from_extended_x(x, Nr)
denominator = x(Nr + 1);
if abs(denominator) < eps
    denominator = 1;
end
v = project_unit_modulus(x(1:Nr) ./ denominator);
end

function v = project_unit_modulus(z)
v = exp(1j .* angle(z));
zeroMask = abs(z) < eps;
v(zeroMask) = 1;
end
