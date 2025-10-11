%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform

    @Acknowledgement: https://mathworks.com/help/phased/ug/waveform-design-for-a-dual-function-mimo-radcom-system.html
    @Dependencies:    Phased Array System Toolbox, Communications Toolbox, Signal Processing Toolbox, DSP System Toolbox
%}

clc;
close all;

%% Warning
% warning('Have you installed following Toolbox: [Phased Array System], [Communications], [Signal Processing], [DSP System]?');
% disp('Press any key to continue if installed...');
% pause;

%% For the reproducibility of experimental results
rng(2024);      % Year of 2024! is this a good reason to use 2024 here?

%% Dependencies
% Externel
addpath('.\Externals');
% Internel
addpath('.\Utils');

%% Load Scenario
isReloadScenario = 0;           % Change to 0 after loading scenario because very time-consuming to load it
if isReloadScenario
    clear all;
    LoadScenario;
    disp('Scenario Loaded!');
    return;
elseif ~exist('HRef', 'var')
    error('main :: Please load simulation scenario first!');
end

%% Sensing-Centric Waveform Design for ISAC
% Note: Nominal channel is fixed after channel estimation. Throughtout waveform designs, we cannot change it.
HBar = HRef + ((2*rand(size(HRef)) - 1) + (2*rand(size(HRef)) - 1) * 1j)*epsilon;

F = chol(Rs, 'lower');              % to ensure R = F*F'

% Specify Design Type
DesignType = 'SensingCentric';      % 'SensingCentric', 'Joint'
% Unify to uppercase
DesignType = upper(DesignType);

switch DesignType
    case 'SENSINGCENTRIC'           % SensingCentric
        disp('Sensing-Centric Waveform Design Using Nominal Channel...');
        DesignMethod = 2;           % '1' (Method 1); '2' (Method 2)
        WaveformDesign_SensingCentric;
    case 'JOINT'                    % Joint
        disp('Joint Waveform Design Using Nominal Channel...');
        PowerType  = 'TPC';         % 'TPC' (Total Power Constraint), 'PAPC' (Per-Antenna Power Constraint)
                                    % NB: 'PAPC' is computationally heavy; be cautious when using it unless you have a better replacement                       
        WaveformDesign_Joint;       % Method 3
    otherwise
        error('main :: Error in DesignType :: Non-exist!')
end

%--------- End of main.m ---------%



