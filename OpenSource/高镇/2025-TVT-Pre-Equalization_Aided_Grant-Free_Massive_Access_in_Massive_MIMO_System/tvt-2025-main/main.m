%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code packet constructs simulations for pre-equlization aided massive 
% access scheme for massive MIMO systems. 
% 
% Coded by: Yueqing Wang, Beijing Institute of Technology
% Email: 17710820190@163.com
% Last change: 01/11/23
%
% ChangeLog:
%
% 2.0 
% 2.1 Add power control
% 2.2 Power control is revised
% 2.3 Add coherent detection as baseline algorithm
% Version 2.3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Initializations
% Clear the workspace etc
clc; clear; close all; 

% Get absolute path to the folder containing this file
basePath = [fileparts(mfilename('fullpath')) filesep];

% Add paths for the required files
addpath([basePath 'algorithms'])
addpath([basePath 'schemes'])
%addpath([basePath 'matlab_parpool'])
addpath([basePath 'mimo_channel_model'])
addpath([basePath 'misc'])

parIn = get_system_parameters();
% variable parameters
varList =[52,54,56,58,60];

ADEP = zeros(numel(varList),1);
BER = zeros(numel(varList),parIn.numAlgIters);
NMSE_CE = zeros(numel(varList),parIn.numAlgIters); 
snr_preeq = 0;
snr_average = 0;

for idxVar = 1:numel(varList)
    % different length of spreading codes
    parIn.lenSprCodes = varList(idxVar);
    for idxSim = 1:parIn.numSim
   
        sample = gen_system_model(parIn);
        
        snr_preeq = snr_preeq + sample.snr_preeq;
        snr_average = snr_average + sample.snr_average; 
        result = mimo_detection(parIn,sample);

        metrics = performance_evaluation(parIn,sample,result);

        ADEP(idxVar) = ADEP(idxVar) + metrics.ADEP;
        BER(idxVar,:) = BER(idxVar,:) + metrics.BER;
        NMSE_CE(idxVar,:) = NMSE_CE(idxVar,:) + metrics.NMSE_CE;
        
        % pirnt the performance of each iterations
        for idxIter = 1:parIn.numAlgIters
        fprintf('sim = %4d,  M = %d, T = %d, ADEP = %6.7f, BER = %6.7f, NMSE = %6.7f, BER_Coarse = %6.7f \n', ...
                 idxSim, parIn.lenSprCodes, parIn.numOFDMSymbols, metrics.ADEP, metrics.BER(idxIter), 10*log10(metrics.NMSE_CE(idxIter)),metrics.BERCoarse);
        fprintf('Average:    M = %d, T = %d, ADEP = %6.7f, BER = %6.7f, NMSE = %6.7f \n\n', ...
                 parIn.lenSprCodes, parIn.numOFDMSymbols, ADEP(idxVar)/idxSim, BER(idxVar,idxIter)/idxSim,10*log10(NMSE_CE(idxVar,idxIter)/idxSim));
        end
    end
end