function [gain, Heff, Phi] = compute_path_gain(Hsr, Hrd, v)
%COMPUTE_PATH_GAIN Compute the executable RIS path-gain objective.
%
%   Inputs:
%       Hsr - Source/Radar-to-RIS channel, size Nr x Nt.
%       Hrd - Stage-2 RIS-domain target/return effective channel, size Nr x Nr.
%       v   - RIS phase vector, size Nr x 1. Expected to satisfy |v_i| = 1
%             when used as a physical RIS phase vector.
%
%   Outputs:
%       gain - Path gain under the current code convention:
%              gain = ||Heff||_F^2, unitless linear power gain.
%       Heff - Equivalent channel, size Nt x Nt:
%              Heff = Hsr' * Phi * Hrd * Phi' * Hsr.
%       Phi  - RIS phase matrix diag(v), size Nr x Nr.
%
%   Notes:
%       This is the Stage-3 executable objective. It is dimensionally
%       consistent with the Stage-2 convention, but it is not claimed to be
%       the exact quadratic T-matrix objective printed in the paper.

arguments
    Hsr {mustBeNumeric}
    Hrd {mustBeNumeric}
    v {mustBeNumeric}
end

v = v(:);
Nr = size(Hsr, 1);

if numel(v) ~= Nr
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "v must contain Nr entries. Got numel(v)=%d and Nr=%d.", numel(v), Nr);
end
if size(Hrd, 1) ~= Nr || size(Hrd, 2) ~= Nr
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "Hrd must be Nr x Nr. Got Hrd %s and Nr=%d.", mat2str(size(Hrd)), Nr);
end

[Heff, Phi] = compute_effective_channel(Hsr, Hrd, v);
gain = norm(Heff, "fro")^2;
end
