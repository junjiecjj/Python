%% MAIN_FIG6_RANGE_DOPPLER_MAPS
% Planned reproduction of Fig. 6: ADMM/CD range-Doppler maps for RIS sizes.
% Current round: placeholder only.

clear; clc;
projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params(); %#ok<NASGU>
error("RIS_MIMO_FMCW:NotImplemented", ...
    "Fig. 6 reproduction is not implemented in the scaffold round.");
