function [B, info] = design_precoder_zf(Heff, txPower)
%DESIGN_PRECODER_ZF Design and normalize the ZF precoder.
%
%   Inputs:
%       Heff    - Effective channel matrix. For stage 2, size Nt x Nt.
%       txPower - Linear transmit power in Watts, or a params struct with
%                 field params.power.txPower_W.
%
%   Outputs:
%       B    - ZF precoder, size size(Heff,2) x size(Heff,1), normalized so
%              that ||B||_F^2 <= txPower. This implementation uses the full
%              power budget when the raw pseudo-inverse is nonzero.
%       info - Struct with raw power, normalized power, scaling factor, and
%              ZF residual metrics.
%
%   Units:
%       txPower must be linear Watts, not dBm.

arguments
    Heff {mustBeNumeric}
    txPower
end

if isstruct(txPower)
    txPower_W = txPower.power.txPower_W;
else
    txPower_W = txPower;
end
validateattributes(txPower_W, {'numeric'}, {'scalar', 'real', 'positive'});

Braw = pinv(Heff);
rawPower_W = norm(Braw, "fro")^2;

if rawPower_W <= eps
    error("RIS_MIMO_FMCW:DegeneratePrecoder", ...
        "The pseudo-inverse has near-zero Frobenius power.");
end

scale = sqrt(txPower_W / rawPower_W);
B = scale .* Braw;
normalizedPower_W = norm(B, "fro")^2;

identityTarget = eye(size(Heff, 1), class(Heff));
rawResidual = Heff * Braw - identityTarget;
normalizedResidual = Heff * B - scale .* identityTarget;

info = struct();
info.txPower_W = txPower_W;
info.rawPower_W = rawPower_W;
info.normalizedPower_W = normalizedPower_W;
info.scale = scale;
info.rawErrorFro = norm(rawResidual, "fro");
info.rawRelativeError = norm(rawResidual, "fro") / max(norm(identityTarget, "fro"), eps);
info.normalizedErrorFro = norm(normalizedResidual, "fro");
info.relativeError = norm(normalizedResidual, "fro") / max(norm(scale .* identityTarget, "fro"), eps);
info.conditionNumber = cond(Heff);
info.rank = rank(Heff);
info.dimension.Heff = size(Heff);
info.dimension.B = size(B);
end
