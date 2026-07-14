function [detectionMask, thresholdPower, noiseEstimatePower, meta] = ca_cfar_2d(rdPower, options)
%CA_CFAR_2D Apply two-dimensional cell-averaging CFAR to an RD power map.
%
%   Inputs:
%       rdPower - Nonnegative linear RD power map, size Nrange x Ndoppler.
%                 Do not pass dB magnitudes to this function.
%       options - Struct fields:
%                 trainingCells [Tr Tv], nonnegative integer cells on each
%                 side of the guard window in range and Doppler dimensions.
%                 guardCells [Gr Gv], nonnegative integer protected cells
%                 on each side of the CUT.
%                 pfa, scalar false-alarm probability in (0, 1).
%
%   Outputs:
%       detectionMask      - Logical CA-CFAR detections, size rdPower.
%       thresholdPower     - Linear CFAR threshold map, size rdPower.
%                            Edge cells without a complete window are NaN.
%       noiseEstimatePower - Linear training-cell mean map, size rdPower.
%                            Edge cells without a complete window are NaN.
%       meta               - Window sizes, training-cell count, threshold
%                            scale alpha, valid-cell mask, and pfa.
%
%   Stage 4 note:
%       CA-CFAR is applied in the linear power domain. The present function
%       assumes homogeneous training statistics around each cell under test.

arguments
    rdPower {mustBeNumeric, mustBeNonnegative}
    options struct = struct()
end

trainingCells = get_pair_option(options, "trainingCells", [6, 6]);
guardCells = get_pair_option(options, "guardCells", [2, 2]);
pfa = get_scalar_option(options, "pfa", 1e-5);
if ~(isscalar(pfa) && isfinite(pfa) && pfa > 0 && pfa < 1)
    error("RIS_MIMO_FMCW:InvalidPfa", "CA-CFAR pfa must be a scalar in (0, 1).");
end

Tr = trainingCells(1);
Tv = trainingCells(2);
Gr = guardCells(1);
Gv = guardCells(2);
outerHalfSize = [Tr + Gr, Tv + Gv];
outerSize = 2 .* outerHalfSize + 1;
guardSize = 2 .* guardCells + 1;
numOuterCells = prod(outerSize);
numGuardCells = prod(guardSize);
numTrainingCells = numOuterCells - numGuardCells;
if numTrainingCells <= 0
    error("RIS_MIMO_FMCW:InvalidCfarWindow", "CA-CFAR requires at least one training cell.");
end

outerKernel = ones(outerSize);
guardKernel = zeros(outerSize);
rangeGuardIdx = Tr + (1:guardSize(1));
dopplerGuardIdx = Tv + (1:guardSize(2));
guardKernel(rangeGuardIdx, dopplerGuardIdx) = 1;
trainingKernel = outerKernel - guardKernel;

trainingSum = conv2(double(rdPower), trainingKernel, "same");
noiseMean = trainingSum ./ numTrainingCells;
alpha = numTrainingCells .* (pfa.^(-1 ./ numTrainingCells) - 1);
threshold = alpha .* noiseMean;

validMask = false(size(rdPower));
rangeValidIdx = (1 + outerHalfSize(1)):(size(rdPower, 1) - outerHalfSize(1));
dopplerValidIdx = (1 + outerHalfSize(2)):(size(rdPower, 2) - outerHalfSize(2));
if ~isempty(rangeValidIdx) && ~isempty(dopplerValidIdx)
    validMask(rangeValidIdx, dopplerValidIdx) = true;
end

detectionMask = validMask & rdPower > threshold;
thresholdPower = nan(size(rdPower));
noiseEstimatePower = nan(size(rdPower));
thresholdPower(validMask) = threshold(validMask);
noiseEstimatePower(validMask) = noiseMean(validMask);

meta = struct();
meta.trainingCells = trainingCells;
meta.guardCells = guardCells;
meta.outerSize = outerSize;
meta.numTrainingCells = numTrainingCells;
meta.pfa = pfa;
meta.alpha = alpha;
meta.validMask = validMask;
end

function pair = get_pair_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    pair = double(options.(fieldName));
else
    pair = defaultValue;
end
pair = pair(:).';
if numel(pair) ~= 2 || any(~isfinite(pair)) || any(pair < 0) || any(pair ~= round(pair))
    error("RIS_MIMO_FMCW:InvalidCfarWindow", ...
        "%s must contain two nonnegative integer cell counts.", fieldName);
end
end

function value = get_scalar_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = options.(fieldName);
else
    value = defaultValue;
end
end
