function [objectiveValue, metrics] = evaluate_ris_objective(Hsr, Hrd, v, params, objectiveType, options)
%EVALUATE_RIS_OBJECTIVE Unified RIS phase objective evaluator.
%
%   Inputs:
%       Hsr           - Source/Radar-to-RIS channel, size Nr x Nt.
%       Hrd           - RIS-domain target/return effective channel, size Nr x Nr.
%       v             - RIS phase vector, size Nr x 1. Physical phases should
%                       satisfy |v_i| = 1.
%       params        - Struct from config/paper_params.m.
%       objectiveType - "path_gain", "zf_snr", or
%                       "zf_snr_with_condition_penalty".
%       options       - Optional struct. For condition-penalty objective,
%                       options.conditionPenaltyAlpha defaults to 0.05.
%
%   Outputs:
%       objectiveValue - Scalar objective being optimized.
%       metrics        - Struct containing Heff, pathGain, snrLinear, snrDb,
%                        condHeff, zfRawPower, zfNormalizedPower,
%                        unitModulusError, and objectiveType.
%
%   Units:
%       SNR is linear in snrLinear and dB in snrDb. Power fields are Watts.

arguments
    Hsr {mustBeNumeric}
    Hrd {mustBeNumeric}
    v {mustBeNumeric}
    params struct
    objectiveType {mustBeTextScalar} = "zf_snr"
    options struct = struct()
end

objectiveType = string(objectiveType);
v = v(:);

[pathGain, Heff, Phi] = compute_path_gain(Hsr, Hrd, v);
[B, zfInfo] = design_precoder_zf(Heff, params.power.txPower_W);
[snrLinear, snrDb] = compute_snr(Heff, B, params.power.noisePower_W);
condHeff = cond(Heff);
unitModulusError = max(abs(abs(v) - 1));

alpha = get_option(options, "conditionPenaltyAlpha", 0.05);
switch objectiveType
    case "path_gain"
        objectiveValue = pathGain;
    case "zf_snr"
        objectiveValue = snrLinear;
    case "zf_snr_with_condition_penalty"
        penalty = 1 + alpha .* log10(max(condHeff, 1)).^2;
        objectiveValue = snrLinear ./ penalty;
    otherwise
        error("RIS_MIMO_FMCW:UnsupportedObjective", ...
            "Unsupported objectiveType: %s.", objectiveType);
end

metrics = struct();
metrics.objectiveType = objectiveType;
metrics.objectiveValue = objectiveValue;
metrics.Heff = Heff;
metrics.Phi = Phi;
metrics.pathGain = pathGain;
metrics.snrLinear = snrLinear;
metrics.snrDb = snrDb;
metrics.condHeff = condHeff;
metrics.zfRawPower = zfInfo.rawPower_W;
metrics.zfNormalizedPower = zfInfo.normalizedPower_W;
metrics.unitModulusError = unitModulusError;
metrics.zfInfo = zfInfo;
end

function value = get_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = options.(fieldName);
else
    value = defaultValue;
end
end
