classdef test_stage4_cfar_detection < matlab.unittest.TestCase
    %TEST_STAGE4_CFAR_DETECTION Unit tests for Stage 4 CA-CFAR RD detection.

    methods (Test)
        function caCfarDetectsStrongCellInNoise(testCase)
            projectRoot = fileparts(fileparts(mfilename("fullpath")));
            addpath(fullfile(projectRoot, "functions"));

            rng(37, "twister");
            rdPower = abs(randn(40, 48) + 1j .* randn(40, 48)).^2;
            targetIdx = [21, 25];
            rdPower(targetIdx(1), targetIdx(2)) = 2e3;

            options = struct("trainingCells", [4, 4], "guardCells", [1, 1], "pfa", 1e-4);
            [detectionMask, thresholdPower, noiseEstimatePower, meta] = ca_cfar_2d(rdPower, options);

            testCase.verifyTrue(detectionMask(targetIdx(1), targetIdx(2)));
            testCase.verifyGreaterThan(rdPower(targetIdx(1), targetIdx(2)), thresholdPower(targetIdx(1), targetIdx(2)));
            testCase.verifyGreaterThan(noiseEstimatePower(targetIdx(1), targetIdx(2)), 0);
            testCase.verifyGreaterThan(meta.numTrainingCells, 0);
        end

        function fullMapCfarAssociatesTruthNeighborhoodPeaks(testCase)
            projectRoot = fileparts(fileparts(mfilename("fullpath")));
            addpath(fullfile(projectRoot, "functions"));

            rangeAxis = (0:0.5:20).';
            velocityAxis = -3:0.25:3;
            rdPower = ones(numel(rangeAxis), numel(velocityAxis));
            rdPower(17, 9) = 8e2;
            rdPower(31, 18) = 1.2e3;
            rdPower(22, 13) = 9e2; % CFAR candidate outside target association windows.
            rdComplex = sqrt(rdPower);

            targets = struct();
            targets.range_m = [rangeAxis(17); rangeAxis(31)];
            targets.velocity_mps = [velocityAxis(9); velocityAxis(18)];

            options = struct();
            options.trainingCells = [3, 3];
            options.guardCells = [1, 1];
            options.pfa = 1e-3;
            options.associationRangeHalfWidth_m = 0.75;
            options.associationVelocityHalfWidth_mps = 0.35;
            detection = detect_rd_targets_cfar(rdComplex, rangeAxis, velocityAxis, targets, options);

            testCase.verifyEqual(detection.numAssociatedTargets, 2);
            testCase.verifyEqual(detection.hit(:), true(2, 1));
            testCase.verifyEqual(detection.peakRange_m(:), targets.range_m(:), "AbsTol", 1e-12);
            testCase.verifyEqual(detection.peakVelocity_mps(:), targets.velocity_mps(:), "AbsTol", 1e-12);
            testCase.verifyGreaterThanOrEqual(detection.numCfarPeaks, 3);
        end
    end
end
