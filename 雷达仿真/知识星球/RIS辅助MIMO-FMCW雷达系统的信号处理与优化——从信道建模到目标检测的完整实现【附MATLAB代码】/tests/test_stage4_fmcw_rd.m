classdef test_stage4_fmcw_rd < matlab.unittest.TestCase
    %TEST_STAGE4_FMCW_RD Stage 4 FMCW echo and RD FFT smoke tests.

    methods (Test)
        function singleTargetPeakIsNearTruth(testCase)
            projectRoot = fileparts(fileparts(mfilename("fullpath")));
            addpath(fullfile(projectRoot, "config"));
            addpath(fullfile(projectRoot, "functions"));

            params = paper_params();
            targets = struct();
            targets.range_m = 25;
            targets.velocity_mps = 3;
            targets.alpha = 1;

            [Y, echoMeta] = generate_fmcw_echo(params, targets, 1, 1e-12);
            [~, RDdB, rangeAxis, velocityAxis] = range_doppler_fft(Y, params);

            [~, peakIdx] = max(RDdB(:));
            [rangeIdx, velocityIdx] = ind2sub(size(RDdB), peakIdx);

            testCase.verifySize(Y, [params.radar.numFastTimeSamples, params.radar.numChirps]);
            testCase.verifyLessThanOrEqual(abs(rangeAxis(rangeIdx) - targets.range_m), ...
                echoMeta.rangeResolution_m);
            testCase.verifyLessThanOrEqual(abs(velocityAxis(velocityIdx) - targets.velocity_mps), ...
                echoMeta.velocityResolution_mps);
        end
    end
end
