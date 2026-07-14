%% MAIN_FIG4_SNR_VS_POWER
% Planned reproduction of Fig. 4: ADMM/CD SNR versus transmit power.
% Current round: placeholder only.

clear; clc;
projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params(); %#ok<NASGU>
error("RIS_MIMO_FMCW:NotImplemented", ...
    "Fig. 4 reproduction is not implemented in the scaffold round.");
