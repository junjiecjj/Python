% close all
% clc
clearvars -except FlagsArray KFlags4Plot CurFlagInd
%%
rng(130,'twister');

%% Creating dirs
if ~exist('McMatFiles','dir')
    mkdir('McMatFiles');
end
%% Functions
RoundFunc          = @(x) ( round(x*10)/10 ) ;
%% Plot Params
linewd                  = 0.8;
hcfontsize              = 20;
MarkerSize              = 9;

%%
tic
%% Which figures to plot - only one flag should have a logical '1' value
PlogFigs_1_2_5      = FlagsArray(1);
PlogFigs_3_4_9      = FlagsArray(2); %
PlotFigs_7          = FlagsArray(3);
PlotFigs_8          = 0;
PlotFigs_6SNR       = 0;
PlotFigs_6Beta      = 0;
%%
if PlogFigs_1_2_5
    KScenarios      = 1;
    MonteCarloFlag  = 0;
    ScenariosList   = [1 12];
end
if PlogFigs_3_4_9  
    KScenarios      = 200;
    MonteCarloFlag  = 0;
    ScenariosList   = [1 12 13];
end
if PlotFigs_7  
    KScenarios      = 200;
    MonteCarloFlag  = 0;
    ScenariosList   = 27;
end
if PlotFigs_8  
    KScenarios      = 200;
    MonteCarloFlag  = 0;
    ScenariosList   = [20 21];
end
if PlotFigs_6SNR     
    KScenarios   	= 200;
    MonteCarloFlag  = 1;
    IsMCforBeta     = 0;
    ScenariosList   = [1 12 14];
end
if PlotFigs_6Beta     
    KScenarios   	= 200;
    MonteCarloFlag  = 1;
    IsMCforBeta     = 1;
    ScenariosList   = [1 12 14];
end
%%
for Mscen = ScenariosList 
    Mscen
    rng(130,'twister'); 
    switch Mscen
        case 1 % Two interference sources -6[dB] input SIR
            KInterferenceSources                = 2;
            InterferenceAmpGain                 = 2+0*rand(1,KInterferenceSources);
            TargetGain2                         = 0;
            KLargeFrames                        = KInterferenceSources;%
        case 12 % Two interference sources -10[dB] input SIR
            KInterferenceSources                = 2;
            InterferenceAmpGain                 = sqrt(10)+0*rand(1,KInterferenceSources);
            TargetGain2                         = 0;
            clear SigInterAtArrayMat SigForGammaTimeDomain SigForGammaStftFreqBin SigInterAtArrMatFft SigInterAtArrayMat
        case 13 % Two interference sources -20[dB] input SIR
            KInterferenceSources                = 2;
            InterferenceAmpGain                 = 10+0*rand(1,KInterferenceSources);
            TargetGain2                         = 0;
            KLargeFrames                        = KInterferenceSources;%
            clear SigInterAtArrayMat SigForGammaTimeDomain SigForGammaStftFreqBin SigInterAtArrMatFft SigInterAtArrayMat hInterToArrTens InterSigZeroPad
        case 14 % Two interference sources 0[dB] input SIR
            KInterferenceSources                = 2;
            InterferenceAmpGain                 = 1;
            TargetGain2                         = 0;
            KLargeFrames                        = KInterferenceSources;%
            clear SigInterAtArrayMat SigForGammaTimeDomain SigForGammaStftFreqBin SigInterAtArrMatFft SigInterAtArrayMat hInterToArrTens InterSigZeroPad
        case 20 % Two desired sources -6[dB] input SIR
            KInterferenceSources                = 2;
            InterferenceAmpGain                 = 2+0*rand(1,KInterferenceSources);
            TargetGain2                         = 1;
            KLargeFrames                        = KInterferenceSources;%
        case 21 % Two desired sources -10[dB] input SIR
            KInterferenceSources                = 2;
            InterferenceAmpGain                 = sqrt(10)+0*rand(1,KInterferenceSources);
            TargetGain2                         = 1;
            KLargeFrames                        = KInterferenceSources;%
            clear SigInterAtArrayMat SigInterAtArrayMat InterSigZeroPad SigForGammaTimeDomain SigForGammaStftFreqBin SigInterAtArrMatFft SigInterAtArrayMat  hInterToArrTens InterSigZeroPad
        case 27 % 14 interference sources, 10 segments, -6dB input SIR
            KInterferenceSources                = 14;
            InterferenceAmpGain                 = 2+0*rand(1,KInterferenceSources);
            TargetGain2                         = 0;
            KLargeFrames                        = 10;
            clear SigInterAtArrayMat SigInterAtArrayMat InterSigZeroPad SigForGammaTimeDomain SigForGammaStftFreqBin SigInterAtArrMatFft SigInterAtArrayMat  hInterToArrTens InterSigZeroPad
    end
    for scen=1:KScenarios
        scen
        if ~MonteCarloFlag             
            InterferenceRejectionCode;              
        else
            Seed = randi(1e5,1,1);
            rng(Seed,'twister');
            InterferenceRejectionCodeMonteCarlo;
            SirMeanMLR_MC(scen,:,:)                 = mean(10.^(SinrMlR/10),2);
            SirMeanMLE_MC(scen,:,:)                 = mean(10.^(SinrMlE/10),2);
            continue;
        end
        
        DirectivityMLESave(scen,:)                  = DirectivityMLE; % linear scale
        DirectivityMLRSave(scen,:)                  = DirectivityMLR;
        DirectivityMvdrESave(scen,:)                = DirectivityMvdrE; % linear scale
        DirectivityMvdrRSave(scen,:)                = DirectivityMvdrR;
        DirectivityMusicESaveOracle(scen,:)         = DirectivityMusicEOracle;
        DirectivityMusicRSaveOracle(scen,:)         = DirectivityMusicROracle;
        DirectivityIntersectSaveOracle(scen,:)      = DirectivityIntersectOracle;
        
        DirectivityMusicESave(scen,:)               = DirectivityMusicE;
        DirectivityMusicRSave(scen,:)               = DirectivityMusicR;
        DirectivityIntersectSave(scen,:)            = DirectivityIntersect;
        
        SirMeanMLR(scen,:)                          = mean(10.^(SinrMlR/10),2);
        SirMeanMLE(scen,:)                          = mean(10.^(SinrMlE/10),2);
        SirMeanMvdrR(scen,:)                        = mean(10.^(SinrMvdrR/10),2);
        SirMeanMvdrE(scen,:)                        = mean(10.^(SinrMvdrE/10),2);
        SirMeanIntersectOracle(scen,:)              = mean(10.^(SinrIntersectOracle/10),2);
        SirMeanMusicROracle(scen,:)                 = mean(10.^(SinrMusicROracle/10),2);
        SirMeanMusicEOracle(scen,:)                 = mean(10.^(SinrMusicEOracle/10),2);
        SirMeanIntersect(scen,:)                    = mean(10.^(SinrIntersect/10),2);
        SirMeanMusicR(scen,:)                       = mean(10.^(SinrMusicR/10),2);
        SirMeanMusicE(scen,:)                       = mean(10.^(SinrMusicE/10),2);
        
        SirMinMLR(scen,:)                           = 10.^(min(SinrMlR,[],2)/10);
        SirMinMLE(scen,:)                           = 10.^(min(SinrMlE,[],2)/10);
        SirMinSinrIntersectOracle(scen,:)           = 10.^(min(SinrIntersectOracle,[],2)/10);
        SirMinMusicROracle(scen,:)                  = 10.^(min(SinrMusicROracle,[],2)/10);
        SirMinMusicEOracle(scen,:)                  = 10.^(min(SinrMusicEOracle,[],2)/10);
        SirMinSinrIntersect(scen,:)                 = 10.^(min(SinrIntersect,[],2)/10);
        SirMinMusicR(scen,:)                        = 10.^(min(SinrMusicR,[],2)/10);
        SirMinMusicE(scen,:)                        = 10.^(min(SinrMusicE,[],2)/10);
        
        SirMaxMLR(scen,:)                           = 10.^(max(SinrMlR,[],2)/10);
        SirMaxMLE(scen,:)                           = 10.^(max(SinrMlE,[],2)/10);
        SirMaxIntersect(scen,:)                     = 10.^(max(SinrIntersect,[],2)/10);
        SirMaxMusicR(scen,:)                        = 10.^(max(SinrMusicR,[],2)/10);
        SirMaxMusicE(scen,:)                        = 10.^(max(SinrMusicE,[],2)/10);
        
        clear DirectivityMLE DirectivityMLR DirectivityMusicE DirectivityMusicR DirectivityIntersect SinrMlR SinrMlE SinrIntersect SinrMusicR SinrMusicE
    end %Scen
    TimeDur = toc/60; %[min]
    %% MonteCarlo
    if MonteCarloFlag
        if ~exist('McMatFiles','dir')
%             mkdir([pwd '\McMatFiles']);
             mkdir('McMatFiles');
        end
        BlueColDef = [0 0.4470 0.7410];
        OrangeColDef = [0.9290    0.6940    0.1250];
        RedColDef = [0.8500    0.3250    0.0980];
        
        MeanSirMeanMLR_MC = squeeze(mean(SirMeanMLR_MC,1));
        MeanSirMeanMLE_MC = squeeze(mean(SirMeanMLE_MC,1));
        if IsMCforBeta
            Str_R = fullfile(pwd,'McMatFiles',['MeanSirMeanMLR_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Beta_KScen' num2str(KScenarios)]);
            Str_E = fullfile(pwd,'McMatFiles',['MeanSirMeanMLE_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Beta_KScen' num2str(KScenarios)]);
            save(Str_R,'MeanSirMeanMLR_MC');
            save(Str_E,'MeanSirMeanMLE_MC');                        
%             save([pwd '\McMatFiles\MeanSirMeanMLR_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Beta_KScen' num2str(KScenarios)],'MeanSirMeanMLR_MC');
%             save([pwd '\McMatFiles\MeanSirMeanMLE_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Beta_KScen' num2str(KScenarios)],'MeanSirMeanMLE_MC');                        
        else
            Str_R = fullfile(pwd,'McMatFiles',['MeanSirMeanMLR_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Epsilon_KScen' num2str(KScenarios)]);
            Str_E = fullfile(pwd,'McMatFiles',['MeanSirMeanMLE_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Epsilon_KScen' num2str(KScenarios)]);
            save(Str_R,'MeanSirMeanMLR_MC');
            save(Str_E,'MeanSirMeanMLE_MC');                    
%             save([pwd '\McMatFiles\MeanSirMeanMLR_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Epsilon_KScen' num2str(KScenarios)],'MeanSirMeanMLR_MC');
%             save([pwd '\McMatFiles\MeanSirMeanMLE_MC_Sir'  num2str(InterferenceAmpGain(1)^2) 'SNR' num2str(Snr_dB) '_Epsilon_KScen' num2str(KScenarios)],'MeanSirMeanMLE_MC');        
        end 
    end
    %% Printing to file
    if ~MonteCarloFlag
        SaveVars;
    end
end %Mscen
if ~MonteCarloFlag   
    if ~PlogFigs_1_2_5
        PlotMain;
    end
else
    PlotMainMonteCarlo;
end
