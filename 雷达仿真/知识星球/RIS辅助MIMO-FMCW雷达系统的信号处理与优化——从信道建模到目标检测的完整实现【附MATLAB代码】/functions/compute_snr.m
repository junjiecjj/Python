function [snrLinear, snrDb] = compute_snr(Heff, B, noisePower)
%COMPUTE_SNR Compute SNR from effective channel, precoder, and noise power.
%
%   Inputs:
%       Heff       - Effective channel matrix. For stage 2, size Nt x Nt.
%       B          - Precoder matrix. For stage 2, size Nt x Nt.
%       noisePower - Linear noise power sigma^2 in Watts. Do not pass dBm.
%
%   Outputs:
%       snrLinear - Linear SNR = ||Heff * B||_F^2 / noisePower.
%       snrDb     - SNR in dB, computed as 10*log10(snrLinear).

arguments
    Heff {mustBeNumeric}
    B {mustBeNumeric}
    noisePower (1,1) double {mustBePositive}
end

if size(Heff, 2) ~= size(B, 1)
    error("RIS_MIMO_FMCW:DimensionMismatch", ...
        "Heff columns must match B rows. Got Heff %s and B %s.", ...
        mat2str(size(Heff)), mat2str(size(B)));
end

signalPower = norm(Heff * B, "fro")^2;
snrLinear = signalPower ./ noisePower;
snrDb = 10 .* log10(snrLinear);
end
