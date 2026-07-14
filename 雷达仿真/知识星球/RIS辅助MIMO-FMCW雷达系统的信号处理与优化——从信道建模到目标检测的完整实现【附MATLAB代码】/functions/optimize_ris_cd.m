function [v, Phi, history] = optimize_ris_cd(Hsr, Hrd, params)
%OPTIMIZE_RIS_CD Optimize RIS phases using a coordinate descent baseline.
%   Placeholder only. The exact CD update rule is not fully specified in the
%   paper text and must be recorded as a reproduction assumption before use.

arguments
    Hsr
    Hrd
    params struct
end

unusedInputs = {Hsr, Hrd, params}; %#ok<NASGU>
v = []; %#ok<NASGU>
Phi = []; %#ok<NASGU>
history = struct(); %#ok<NASGU>

error("RIS_MIMO_FMCW:NotImplemented", ...
    "optimize_ris_cd is a placeholder. CD details must be confirmed first.");
end
