classdef test_stage4_pd_vs_snr < matlab.unittest.TestCase
    %TEST_STAGE4_PD_VS_SNR Smoke tests for Stage 4 Pd-vs-SNR experiment.

    methods (Test)
        function quickModeReturnsPdCurvesAndArtifacts(testCase)
            projectRoot = fileparts(fileparts(mfilename("fullpath")));
            addpath(fullfile(projectRoot, "main"));

            options = struct();
            options.numTrials = 2;
            options.echoSnrDb = [-10, 10];
            options.saveOutputs = false;
            options.verbose = false;
            result = main_stage4_pd_vs_snr("quick", options);

            testCase.verifyEqual(result.mode, "quick");
            testCase.verifyEqual(result.echoSnrDb, options.echoSnrDb);
            testCase.verifySize(result.pdPerTarget.noRis, [4, 2]);
            testCase.verifySize(result.pdPerTarget.randomRis, [4, 2]);
            testCase.verifySize(result.pdPerTarget.optimizedRis, [4, 2]);
            testCase.verifySize(result.pdAverage.noRis, [1, 2]);
            testCase.verifySize(result.pdAverage.randomRis, [1, 2]);
            testCase.verifySize(result.pdAverage.optimizedRis, [1, 2]);
            testCase.verifyEqual(result.numTrials, 2);
            testCase.verifyEqual(result.cfarOptions.pfa, 1e-4);
        end
    end
end
