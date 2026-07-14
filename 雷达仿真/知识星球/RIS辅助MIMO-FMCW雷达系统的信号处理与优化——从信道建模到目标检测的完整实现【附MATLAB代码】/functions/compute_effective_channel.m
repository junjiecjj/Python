function [Heff, Phi, meta] = compute_effective_channel(Hsr, Hrd, v)
%COMPUTE_EFFECTIVE_CHANNEL Build the RIS-assisted effective MIMO channel.
%
%   Inputs:
%       Hsr - Radar/RIS channel, size Nr x Nt.
%       Hrd - RIS-domain target/return channel, size Nr x Nr.
%       v   - RIS phase vector, size Nr x 1, unitless, expected |v_i| = 1.
%
%   Outputs:
%       Heff - Effective MIMO channel, size Nt x Nt.
%       Phi  - RIS diagonal phase matrix, size Nr x Nr.
%       meta - Struct with dimensions and unit-modulus error.
%
%   Current engineering convention:
%       Phi = diag(v)
%       Heff = Hsr' * Phi * Hrd * Phi' * Hsr

arguments
    Hsr {mustBeNumeric}
    Hrd {mustBeNumeric}
    v {mustBeNumeric}
end

v = v(:);
Nr = size(Hsr, 1);
if numel(v) ~= Nr
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "v must have Nr elements. Got numel(v)=%d and Nr=%d.", numel(v), Nr);
end
if size(Hrd, 1) ~= Nr || size(Hrd, 2) ~= Nr
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "Hrd must be Nr x Nr. Got Hrd %s and Nr=%d.", mat2str(size(Hrd)), Nr);
end

Phi = diag(v);
Heff = Hsr' * Phi * Hrd * Phi' * Hsr;

meta = struct();
meta.HsrSize = size(Hsr);
meta.HrdSize = size(Hrd);
meta.vSize = size(v);
meta.PhiSize = size(Phi);
meta.HeffSize = size(Heff);
meta.unitModulusError = max(abs(abs(v) - 1));
end
