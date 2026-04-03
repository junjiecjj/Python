function [isl, islr, ml] = compute_islr(x, data)
% COMPUTE_ISLR  Compute ISL, mainlobe power, and normalized ISL.
%
% Inputs:
%   x    : Waveform vector in the frequency domain.
%   data : Struct containing the AF operator and B diagonal entries.
%
% Outputs:
%   isl  : Integrated sidelobe level.
%   islr : Normalized ISL, i.e., ISL / mainlobe power.
%   ml   : Mainlobe power.
%
% Notes:
%   The ISL computation follows the vectorized formulation in the
%   TWC 2025 paper; see the compact expressions around Eqs. (31) and (35a).

s = data.AeH * x;
Y = s * s';
vecY = Y(:);

isl = real(vecY' * (data.bdiag .* vecY));
ml = abs(s' * s)^2;
islr = isl / max(ml, eps);
end
