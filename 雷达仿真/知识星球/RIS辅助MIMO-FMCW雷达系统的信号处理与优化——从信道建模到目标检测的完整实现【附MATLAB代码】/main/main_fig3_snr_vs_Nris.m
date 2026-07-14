%% MAIN_FIG3_SNR_VS_NRIS
% Planned reproduction of Fig. 3: ADMM/CD SNR versus RIS element count.
% Current round: placeholder only.

clear; clc;
projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params(); %#ok<NASGU>
error("RIS_MIMO_FMCW:NotImplemented", ...
    "Fig. 3 reproduction is not implemented in the scaffold round.");
