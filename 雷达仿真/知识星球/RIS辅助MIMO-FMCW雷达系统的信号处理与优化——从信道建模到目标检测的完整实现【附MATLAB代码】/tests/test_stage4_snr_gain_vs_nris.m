classdef test_stage4_snr_gain_vs_nris < matlab.unittest.TestCase
    %TEST_STAGE4_SNR_GAIN_VS_NRIS Smoke tests for dense N_RIS gain sweep.

    methods (Test)
        function reducedSweepReturnsGainMetrics(testCase)
            projectRoot = fileparts(fileparts(mfilename("fullpath")));
            addpath(fullfile(projectRoot, "main"));

            options = struct();
            options.NrisAxis = [4, 8];
            options.numTrials = 1;
            options.saveOutputs = false;
            options.verbose = false;
            result = main_stage4_snr_gain_vs_nris(options);

            testCase.verifyEqual(result.NrisAxis, [4, 8]);
            testCase.verifyEqual(result.numTrials, 1);
            testCase.verifySize(result.raw.randomSnrDb, [1, 2]);
            testCase.verifySize(result.raw.optimizedSnrDb, [1, 2]);
            testCase.verifySize(result.summary.snrGainDbMean, [1, 2]);
            testCase.verifySize(result.summary.runtimeMean_s, [1, 2]);
            testCase.verifyEqual(result.config.phaseGridSize, 16);
        end
    end
end
