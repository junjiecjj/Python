%% MAIN_REPRODUCE_ALL
% One-click entry for all reproduction experiments.
% Current round: project scaffold only; algorithms are not implemented.

clear; clc;
projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "config"));
addpath(fullfile(projectRoot, "functions"));

params = paper_params(); %#ok<NASGU>
error("RIS_MIMO_FMCW:NotImplemented", ...
    "main_reproduce_all is a placeholder. Implement sub-experiments first.");
