function [Hsr, Hrd, meta] = generate_channels(params)
%GENERATE_CHANNELS Generate RIS-assisted radar channel matrices.
%
%   Input:
%       params - Struct from config/paper_params.m.
%
%   Outputs:
%       Hsr  - Source/Radar-to-RIS channel, size Nr x Nt.
%       Hrd  - Stage-2 RIS-domain target/return effective channel, size
%              Nr x Nr. This dimension is chosen so that
%              Heff = Hsr' * Phi * Hrd * Phi' * Hsr is Nt x Nt.
%       meta - Struct recording dimensions, Rician K factor, path loss,
%              geometry assumptions, and random seed.
%
%   Units:
%       Distances are meters. Path loss values are linear power ratios.

arguments
    params struct
end

Nt = params.array.Nt;
Nr = params.array.Nr_default;
K = params.channel.ricianK_linear;
alpha = params.channel.pathLossExponent;
d0 = params.channel.referenceDistance_m;
dSr = params.channel.sourceToRisDistance_m;
dRt = params.channel.risToTargetDistance_m;

if isfield(params.repro, "resetRngInGenerateChannels") && params.repro.resetRngInGenerateChannels
    rng(params.repro.rngSeed, "twister");
end

pathLossSr = (d0 / dSr)^alpha;
pathLossRt = (d0 / dRt)^alpha;

losSr = steering_vector(Nr, 0.23) * steering_vector(Nt, -0.17)';
losRd = steering_vector(Nr, 0.31) * steering_vector(Nr, -0.29)';

nlosSr = complex_gaussian(Nr, Nt);
nlosRd = complex_gaussian(Nr, Nr);

ricianLosWeight = sqrt(K / (K + 1));
ricianNlosWeight = sqrt(1 / (K + 1));

Hsr = sqrt(pathLossSr) .* (ricianLosWeight .* losSr + ricianNlosWeight .* nlosSr);
Hrd = sqrt(pathLossRt) .* (ricianLosWeight .* losRd + ricianNlosWeight .* nlosRd);

meta = struct();
meta.model = params.channel.model;
meta.HsrSize = size(Hsr);
meta.HrdSize = size(Hrd);
meta.Nt = Nt;
meta.Nr = Nr;
meta.ricianK_dB = params.channel.ricianK_dB;
meta.ricianK_linear = K;
meta.pathLossExponent = alpha;
meta.referenceDistance_m = d0;
meta.sourceToRisDistance_m = dSr;
meta.risToTargetDistance_m = dRt;
meta.pathLossSr_linear = pathLossSr;
meta.pathLossRt_linear = pathLossRt;
meta.rngSeed = params.repro.rngSeed;
meta.note = "Hrd is a stage-2 RIS-domain effective target/return channel.";
end

function H = complex_gaussian(numRows, numCols)
%COMPLEX_GAUSSIAN Unit-variance circular complex Gaussian matrix.
H = (randn(numRows, numCols) + 1j .* randn(numRows, numCols)) ./ sqrt(2);
end

function a = steering_vector(numElements, phaseStep)
%STEERING_VECTOR Unit-norm narrowband ULA steering vector.
indices = (0:numElements-1).';
a = exp(1j .* 2 .* pi .* phaseStep .* indices) ./ sqrt(numElements);
end
