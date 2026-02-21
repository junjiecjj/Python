% This code plots Figure 6


%% Loading data
if IsMCforBeta
    MeanSirMeanMLR_MC_Sir0SNR50_Beta = LoadDataMC(['MeanSirMeanMLR_MC_Sir1SNR50_Beta_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLR_MC_Sir6SNR50_Beta = LoadDataMC(['MeanSirMeanMLR_MC_Sir4SNR50_Beta_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLR_MC_Sir10SNR50_Beta = LoadDataMC(['MeanSirMeanMLR_MC_Sir10SNR50_Beta_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLE_MC_Sir0SNR50_Beta = LoadDataMC(['MeanSirMeanMLE_MC_Sir1SNR50_Beta_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLE_MC_Sir6SNR50_Beta = LoadDataMC(['MeanSirMeanMLE_MC_Sir4SNR50_Beta_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLE_MC_Sir10SNR50_Beta = LoadDataMC(['MeanSirMeanMLE_MC_Sir10SNR50_Beta_KScen' num2str(KScenarios)],1);
else
    MeanSirMeanMLE_MC_Sir0SNR50_Epsilon = LoadDataMC(['MeanSirMeanMLE_MC_Sir1SNR50_Epsilon_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLR_MC_Sir0SNR50_Epsilon = LoadDataMC(['MeanSirMeanMLR_MC_Sir1SNR50_Epsilon_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLE_MC_Sir6SNR50_Epsilon = LoadDataMC(['MeanSirMeanMLE_MC_Sir4SNR50_Epsilon_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLR_MC_Sir6SNR50_Epsilon = LoadDataMC(['MeanSirMeanMLR_MC_Sir4SNR50_Epsilon_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLE_MC_Sir10SNR50_Epsilon = LoadDataMC(['MeanSirMeanMLE_MC_Sir10SNR50_Epsilon_KScen' num2str(KScenarios)],1);
    MeanSirMeanMLR_MC_Sir10SNR50_Epsilon = LoadDataMC(['MeanSirMeanMLR_MC_Sir10SNR50_Epsilon_KScen' num2str(KScenarios)],1);
end
%%
if IsMCforBeta
    BetaRiemMAt     = [MeanSirMeanMLR_MC_Sir0SNR50_Beta'     MeanSirMeanMLR_MC_Sir6SNR50_Beta'     MeanSirMeanMLR_MC_Sir10SNR50_Beta' ];
    BetaEucMAt      = [MeanSirMeanMLE_MC_Sir0SNR50_Beta'     MeanSirMeanMLE_MC_Sir6SNR50_Beta'     MeanSirMeanMLE_MC_Sir10SNR50_Beta' ];
else
    EpsilonRiemMAt  = [MeanSirMeanMLR_MC_Sir0SNR50_Epsilon'  MeanSirMeanMLR_MC_Sir6SNR50_Epsilon'  MeanSirMeanMLR_MC_Sir10SNR50_Epsilon' ];
    EpsilonEucMAt   = [MeanSirMeanMLE_MC_Sir0SNR50_Epsilon'  MeanSirMeanMLE_MC_Sir6SNR50_Epsilon'  MeanSirMeanMLE_MC_Sir10SNR50_Epsilon' ];
end
%%
BetaVec         = [0.15 0.2 0.25 0.3 0.35 0.4];
EpsilonVec      = fliplr(10.^(-([20 30 40 50])/20));
%% Plot Params
linewd                  = 0.8;
hcfontsize              = 20;
MarkerSize              = 9;
%%
BlueColDef = [0 0.4470 0.7410];
OrangeColDef = [0.9290    0.6940    0.1250];
RedColDef = [0.8500    0.3250    0.0980];
%%
if IsMCforBeta
    MarkerVec = ['p','o','<','s'];
    AlphaVec = [1 0.8 0.6 0.4];
    
    figure;hold on
    
    ColorBlueMat = [ 0    0.4510    0.7412 ;  0.0980    0.5882    0.8902 ; 0.3020    0.7451    0.9604 ;  0.5137    0.8235    0.9882];
    ColorRedMat  = [0.6196    0.0627    0.1647 ; 0.7882    0.1098    0.2353 ; 0.9216    0.2118    0.3412 ; 0.9608    0.4118    0.5137];
    for b=1:size(BetaRiemMAt,2)
        s =  plot(BetaVec*1e3,squeeze(BetaRiemMAt(:,b)),'Color',BlueColDef,'LineWidth',3-0.4*b,'Marker',MarkerVec(b),'MarkerSize',6,'MarkerFaceColor',BlueColDef);
        s.Color(4) = AlphaVec(b);
    end
    for b=1:size(BetaEucMAt,2)
        s =  plot(BetaVec*1e3,squeeze(BetaEucMAt(:,b)),'Color',RedColDef,'LineWidth',3-0.4*b,'Marker',MarkerVec(b),'MarkerSize',6,'MarkerFaceColor',RedColDef);
        s.Color(4) = AlphaVec(b);
    end    
    legend([[num2str([0 -6 -10]') repmat('[dB] (Riem)',3,1)] ; [num2str([0 -6 -10]') repmat('[dB]  (Euc)',3,1)]]);
    % 
    xlabel('\beta[msec]');
    ylabel('[dB]');
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    box on ; grid on
    saveas(gcf,fullfile('/results', 'Fig6Top_SirForDifferentBeta.jpg'))
else
    MarkerVec = ['p','o','<','s','x'];
    AlphaVec = [1 0.8 0.6 0.4 0.35];
    
    figure;hold on
    
    ColorBlueMat = [ 0    0.4510    0.7412 ;  0.0980    0.5882    0.8902 ; 0.3020    0.7451    0.9604 ;  0.5137    0.8235    0.9882];
    ColorRedMat  = [0.6196    0.0627    0.1647 ; 0.7882    0.1098    0.2353 ; 0.9216    0.2118    0.3412 ; 0.9608    0.4118    0.5137];
    for b=1:size(EpsilonRiemMAt,2)
        s =  plot(20*log10(1./EpsilonVec),squeeze(EpsilonRiemMAt(:,b)),'Color',BlueColDef,'LineWidth',3-0.4*b,'Marker',MarkerVec(b),'MarkerSize',6,'MarkerFaceColor',BlueColDef);
        s.Color(4) = AlphaVec(b);
    end
    for b=1:size(EpsilonRiemMAt,2)
        s =  plot(20*log10(1./EpsilonVec),squeeze(EpsilonEucMAt(:,b)),'Color',RedColDef,'LineWidth',3-0.4*b,'Marker',MarkerVec(b),'MarkerSize',6,'MarkerFaceColor',RedColDef);
        s.Color(4) = AlphaVec(b);
    end 
    Snr4Str = [0 -6 -10]';
    legend([[num2str(Snr4Str) repmat('[dB] (Riem)',length(Snr4Str),1)] ; [num2str(Snr4Str) repmat('[dB]  (Euc)',length(Snr4Str),1)]]);
    %  
    xlabel('SNR[dB]');
    ylabel('[dB]');
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    box on ; grid on
    saveas(gcf,fullfile('/results', 'Fig6Bottom_SirForDifferentSnr.jpg'))
end
%%


%%
function MC_data = LoadDataMC(mFileName,dB_flag)
FolderFileName = fullfile(pwd,'McMatFiles',mFileName);
% FolderName = [pwd '\McMatFiles\'];
% load([FolderName mFileName])
load(FolderFileName)
if dB_flag
    if exist('MeanSirMeanMLE_MC')
        MC_data = 10*log10(mean(MeanSirMeanMLE_MC,1));
    else
        MC_data = 10*log10(mean(MeanSirMeanMLR_MC,1));
    end
else
    if exist('MeanSirMeanMLE_MC')
        MC_data = mean(MeanSirMeanMLE_MC,1);
    else
        MC_data = mean(MeanSirMeanMLR_MC,1);
    end
end
end