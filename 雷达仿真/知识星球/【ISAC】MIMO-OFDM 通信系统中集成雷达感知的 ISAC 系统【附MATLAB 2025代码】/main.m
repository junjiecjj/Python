% Main script for ISAC simulation
% This script reads configuration parameters and runs the ISAC simulation

clear; clc; close all;

script_path = fileparts(mfilename('fullpath'));
cd(script_path);
%% Configuration
params = './configs/config_default.csv'; % Default configuration file name
addpath './functions/helpers/'
addpath './functions/tasks/'
addpath './functions/utils/'
% Show or save figures
SHOW_IMAGES = true;  % Set to false to suppress figure display
SAVE_IMAGES = true;  % Set to false to skip saving figures

% Check if configuration file exists
if ~isfile(params)
    error('Configuration file %s not found. Please create the configuration file.', params);
end

%% Read parameters
if ischar(params) || isstring(params)
    configTable = readtable(params);
    result = cell(height(configTable), 1);
    
    for i = 1:height(configTable)
        fprintf("Running configuration %d/%d...\n", i, height(configTable));
        row = configTable(i, :);
        paramStruct = table2struct(row);
        
        % Attach image options to the paramStruct
        paramStruct.SHOW_IMAGES = SHOW_IMAGES;
        paramStruct.SAVE_IMAGES = SAVE_IMAGES;

        run_isac_simulation(paramStruct);
        
        % close all;
    end
end
