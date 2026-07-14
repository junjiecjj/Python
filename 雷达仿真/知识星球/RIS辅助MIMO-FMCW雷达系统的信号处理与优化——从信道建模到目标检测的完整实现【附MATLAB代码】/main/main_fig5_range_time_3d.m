%% MAIN_FIG5_RANGE_TIME_3D
% Planned reproduction of Fig. 5: multi-target range-time 3D amplitude map.
% Current round: placeholder only.

clear; clc;
projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params(); %#ok<NASGU>
error("RIS_MIMO_FMCW:NotImplemented", ...
    "Fig. 5 reproduction is not implemented in the scaffold round.");
