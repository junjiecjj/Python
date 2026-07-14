function detection = detect_rd_targets_cfar(RD_complex, rangeAxis, velocityAxis, targets, options)
%DETECT_RD_TARGETS_CFAR Detect RD peaks by full-map CA-CFAR and truth association.
%
%   Inputs:
%       RD_complex   - Complex range-Doppler spectrum, size Nrange x Ndoppler.
%       rangeAxis    - Range axis in meters, Nrange x 1 or 1 x Nrange.
%       velocityAxis - Velocity axis in m/s, Ndoppler x 1 or 1 x Ndoppler.
%       targets      - Struct with truth fields range_m and velocity_mps.
%                      Each field is Q x 1 in meters and m/s.
%       options      - Struct fields forwarded to ca_cfar_2d:
%                      trainingCells, guardCells, pfa.
%                      Additional fields:
%                      associationRangeHalfWidth_m,
%                      associationVelocityHalfWidth_mps,
%                      localMaxRadiusCells [range doppler].
%
%   Outputs:
%       detection - Struct containing:
%                   CFAR maps, full-map peak candidates, per-target hit
%                   flags, associated peak values/axes/indices, association
%                   errors, and detection counts.
%
%   Stage 4 note:
%       CFAR detection is performed on the full RD map. Target truth is used
%       only after detection to associate candidate peaks for validation.

arguments
    RD_complex {mustBeNumeric}
    rangeAxis {mustBeNumeric}
    velocityAxis {mustBeNumeric}
    targets struct
    options struct = struct()
end

rangeAxis = rangeAxis(:);
velocityAxis = velocityAxis(:).';
rdPower = abs(RD_complex).^2;
rdDb = 10 .* log10(rdPower + eps);
[cfarMask, thresholdPower, noiseEstimatePower, cfarMeta] = ca_cfar_2d(rdPower, options);
localMaxRadiusCells = get_pair_option(options, "localMaxRadiusCells", [1, 1]);
peakMask = full_map_peak_mask(rdPower, cfarMask, localMaxRadiusCells);
[candidateRangeIdx, candidateVelocityIdx] = find(peakMask);
candidatePower = rdPower(peakMask);
[candidatePower, sortIdx] = sort(candidatePower, "descend");
candidateRangeIdx = candidateRangeIdx(sortIdx);
candidateVelocityIdx = candidateVelocityIdx(sortIdx);

numTargets = numel(targets.range_m);
associationRangeHalfWidth_m = get_scalar_option(options, "associationRangeHalfWidth_m", 1.0);
associationVelocityHalfWidth_mps = get_scalar_option(options, "associationVelocityHalfWidth_mps", 0.5);

detection = struct();
detection.hit = false(numTargets, 1);
detection.peakDb = nan(numTargets, 1);
detection.peakPower = nan(numTargets, 1);
detection.peakRange_m = nan(numTargets, 1);
detection.peakVelocity_mps = nan(numTargets, 1);
detection.rangeError_m = nan(numTargets, 1);
detection.velocityError_mps = nan(numTargets, 1);
detection.rangeIdx = nan(numTargets, 1);
detection.velocityIdx = nan(numTargets, 1);
for targetIdx = 1:numTargets
    if isempty(candidatePower)
        continue;
    end
    candidateRange_m = rangeAxis(candidateRangeIdx);
    candidateVelocity_mps = velocityAxis(candidateVelocityIdx).';
    inAssociationWindow = ...
        abs(candidateRange_m - targets.range_m(targetIdx)) <= associationRangeHalfWidth_m & ...
        abs(candidateVelocity_mps - targets.velocity_mps(targetIdx)) <= associationVelocityHalfWidth_mps;
    if ~any(inAssociationWindow)
        continue;
    end
    candidateIdx = find(inAssociationWindow, 1, "first");
    rangeIdx = candidateRangeIdx(candidateIdx);
    velocityIdx = candidateVelocityIdx(candidateIdx);
    detection.hit(targetIdx) = true;
    detection.peakDb(targetIdx) = rdDb(rangeIdx, velocityIdx);
    detection.peakPower(targetIdx) = rdPower(rangeIdx, velocityIdx);
    detection.peakRange_m(targetIdx) = rangeAxis(rangeIdx);
    detection.peakVelocity_mps(targetIdx) = velocityAxis(velocityIdx);
    detection.rangeError_m(targetIdx) = detection.peakRange_m(targetIdx) - targets.range_m(targetIdx);
    detection.velocityError_mps(targetIdx) = detection.peakVelocity_mps(targetIdx) - targets.velocity_mps(targetIdx);
    detection.rangeIdx(targetIdx) = rangeIdx;
    detection.velocityIdx(targetIdx) = velocityIdx;
end

detection.rdPower = rdPower;
detection.rdDb = rdDb;
detection.cfarMask = cfarMask;
detection.thresholdPower = thresholdPower;
detection.noiseEstimatePower = noiseEstimatePower;
detection.cfarMeta = cfarMeta;
detection.peakMask = peakMask;
detection.candidateRangeIdx = candidateRangeIdx;
detection.candidateVelocityIdx = candidateVelocityIdx;
detection.candidateRange_m = rangeAxis(candidateRangeIdx);
detection.candidateVelocity_mps = velocityAxis(candidateVelocityIdx).';
detection.candidatePower = candidatePower;
detection.candidateDb = 10 .* log10(candidatePower + eps);
detection.numCfarCells = nnz(cfarMask);
detection.numCfarPeaks = numel(candidatePower);
detection.numAssociatedTargets = nnz(detection.hit);
detection.associationRangeHalfWidth_m = associationRangeHalfWidth_m;
detection.associationVelocityHalfWidth_mps = associationVelocityHalfWidth_mps;
end

function peakMask = full_map_peak_mask(rdPower, cfarMask, radiusCells)
peakMask = false(size(cfarMask));
detectedLinearIdx = find(cfarMask);
for idx = 1:numel(detectedLinearIdx)
    [rangeIdx, velocityIdx] = ind2sub(size(cfarMask), detectedLinearIdx(idx));
    rangeWindow = max(1, rangeIdx - radiusCells(1)):min(size(rdPower, 1), rangeIdx + radiusCells(1));
    velocityWindow = max(1, velocityIdx - radiusCells(2)):min(size(rdPower, 2), velocityIdx + radiusCells(2));
    localPower = rdPower(rangeWindow, velocityWindow);
    if rdPower(rangeIdx, velocityIdx) >= max(localPower(:))
        peakMask(rangeIdx, velocityIdx) = true;
    end
end
end

function pair = get_pair_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    pair = double(options.(fieldName));
else
    pair = defaultValue;
end
pair = pair(:).';
if numel(pair) ~= 2 || any(~isfinite(pair)) || any(pair < 0) || any(pair ~= round(pair))
    error("RIS_MIMO_FMCW:InvalidPairOption", ...
        "%s must contain two nonnegative integer cell counts.", fieldName);
end
end

function value = get_scalar_option(options, fieldName, defaultValue)
if isfield(options, fieldName)
    value = double(options.(fieldName));
else
    value = defaultValue;
end
if ~(isscalar(value) && isfinite(value) && value >= 0)
    error("RIS_MIMO_FMCW:InvalidScalarOption", "%s must be a finite nonnegative scalar.", fieldName);
end
end
