function parIn = get_system_parameters()

%% System Parameters
parIn.numUEs = 500;           % number of potential UEs
parIn.numActUEs = 50;        % number of active UEs
parIn.numBSAntennas = 128;   % number of BS antennas for each subarrays
parIn.numUEAntennas = 1;     % number of UE antennas
parIn.numOFDMSymbols = 20;
parIn.lenSprCodes = 60;

% Modulation mode. The QAM has not been supported yet
parIn.modType = 'QPSK';       % 'PSK' or 'QAM'
parIn.modOrder = 4;          % modulation order
parIn.bitsperSymbol = log2(parIn.modOrder);
parIn.bitsperFrame = parIn.numOFDMSymbols * parIn.bitsperSymbol * parIn.numActUEs;

%% Simulation Parameters
%parIn.snrdB = 10;
parIn.numSim = 1000;
%%%%%%%%%%%%
parIn.scheme = 1; % 0 for 'single' 1 for 'mimo' 2 for 'coherent'
%%%%%%%%%%%%

%% Channel Paramters
parIn.Lp_min = 8;  
parIn.Lp_max = 12;
parIn.bandWidth = 1e7;
parIn.fs = 1e7;
parIn.lenCP = 16;
parIn.pathGain = 1;
parIn.channelTh = 0.5;
parIn.angleSpread = 7.5;
parIn.preEqTh = 0.2;

% Path loss, transmit power, and variance of AWGN
% Macro cell, 2 GHz, 3GPP TR 36.931 version 9.0.0 Release 9
% L = 128.1 + 37.6log10(d)
parIn.lossA = 128.1;         % loss in reference distance
parIn.lossB = 37.6;          % loss parameter
parIn.distMin = 0.1;         % min distance in km
parIn.distMax = 1;           % max distance in km
parIn.Ps_dBm = 7;           % transmitted power in dBm, max 23 dBm
parIn.psdNoise = -174;       % dBm/Hz
parIn.PA = 1e3;              % power amplify
parIn.PsMax = 10^(parIn.Ps_dBm/10)*1e-3;

% averaging
parIn.pathLossEp_dBm = parIn.lossA + parIn.lossB*(-0.323183);
parIn.fadingEp = 0.5*(parIn.Lp_min+parIn.Lp_max);
parIn.PowPreEq = 0.2543;
parIn.PowSprCodes = 1;
parIn.Pr = 10^((parIn.Ps_dBm - parIn.pathLossEp_dBm)/10)*1e-3;
if ~parIn.scheme
  parIn.PowSprCodes = 1/parIn.numUEs;
  % parIn.PowSprCodes = 1/parIn.lenSprCodes;
end
parIn.zeta = parIn.Pr/(parIn.PowPreEq*parIn.PowSprCodes*parIn.fadingEp);

parIn.varNoise = 10^(parIn.psdNoise/10)*1e-3*parIn.bandWidth;



%% Algorithm Parameters
parIn.numAlgIters = 3;
% MMV-AMP algorithm
parIn.dampAMPCoarse = 0.3;
parIn.numAMPItersCoarse = 50;
parIn.adThAMP = 0.2;


% GMMV-AMP algorithm
parIn.dampAMP = 0.3;         % damping factor
parIn.numAMPIters = 100;     % number of AMP iterations
parIn.tolAMP = 1e-4;         % stopping criterion
parIn.sparseStr = 1;    % 0/1 'structrued'/'clustered'

% OAMP-MMV-SSL algorithm
parIn.dampOAMP = 0;
parIn.priorOAMP = 1; %0/1 for Gaussian/MPSK
parIn.numOAMPIters = 50;
parIn.adThOAMP = 0.5;

%% Debug Paramters
parIn.debugChannel = false;
parIn.debugSNR = true;


