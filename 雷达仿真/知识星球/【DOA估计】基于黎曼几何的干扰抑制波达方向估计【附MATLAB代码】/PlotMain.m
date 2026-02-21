
%% Plot Params
linewd                  = 0.8;
hcfontsize              = 20;
MarkerSize              = 9;
%%
if PlogFigs_3_4_9
    %%
    SirMeanMLAll = [SirMeanMLRMScen1SNR50(:) SirMeanMLEMScen1SNR50(:) SirMeanMLRMScen12SNR50(:) SirMeanMLEMScen12SNR50(:) ...
        SirMeanMLRMScen13SNR50(:) SirMeanMLEMScen13SNR50(:)];
    DirectivityMLAll = [DirectivityMLRSaveMScen1SNR50(:) DirectivityMLESaveMScen1SNR50(:) DirectivityMLRSaveMScen12SNR50(:) DirectivityMLESaveMScen12SNR50(:) ...
        DirectivityMLRSaveMScen13SNR50(:) DirectivityMLESaveMScen13SNR50(:)];
    
    
    LabelsML1 = {'Riemannian','Euclidean', 'Riemannian' , 'Euclidean', 'Riemannian' , 'Euclidean'};
    LabelsML2 = {'-6[dB]','-6[dB]', '-10[dB]','-10[dB]','-20[dB]','-20[dB]' };
    LabelsAllML = {LabelsML1,LabelsML2};
    
    MyBoxSubPlotFunc(DirectivityMLAll, LabelsAllML, 'Directivity - DS' ,3);
    saveas(gcf,fullfile('/results', 'Fig3Bottom_DirectivityDS.jpg'))    
    MyBoxSubPlotFunc(SirMeanMLAll, LabelsAllML, 'SIR - DS' ,3);
    saveas(gcf,fullfile('/results', 'Fig3Top_SirDS.jpg'))    
    %
    %-------------------------------------------------------------------------------------
    DirectivitySSAll = [DirectivityMusicRSaveMScen1SNR50(:) DirectivityMusicESaveMScen1SNR50(:) DirectivityIntersectSaveMScen1SNR50(:)....
        DirectivityMusicRSaveMScen12SNR50(:) DirectivityMusicESaveMScen12SNR50(:)    DirectivityIntersectSaveMScen12SNR50(:) ...
        DirectivityMusicRSaveMScen13SNR50(:) DirectivityMusicESaveMScen13SNR50(:)    DirectivityIntersectSaveMScen13SNR50(:)];
    LabelsML1 = {'Riemannian','Euclidean','Intersection', 'Riemannian' , 'Euclidean', 'Intersection', 'Riemannian' , 'Euclidean', 'Intersection'};
    LabelsML2 = {'-6[dB]','-6[dB]', '-6[dB]', '-10[dB]','-10[dB]', '-10[dB]','-20[dB]','-20[dB]' ,'-20[dB]'};
    LabelsAllSS = {LabelsML1,LabelsML2};  
    
    
    %-------------------------------------------------------------------------------------
    DirectivityOracleSSAll = [DirectivityMusicRSaveOracleMScen1SNR50(:) DirectivityMusicESaveOracleMScen1SNR50(:) DirectivityIntersectSaveOracleMScen1SNR50(:)....
        DirectivityMusicRSaveOracleMScen12SNR50(:) DirectivityMusicESaveOracleMScen12SNR50(:)    DirectivityIntersectSaveOracleMScen12SNR50(:) ...
        DirectivityMusicRSaveOracleMScen13SNR50(:) DirectivityMusicESaveOracleMScen13SNR50(:)    DirectivityIntersectSaveOracleMScen13SNR50(:)];
    LabelsML1 = {'Riemannian','Euclidean','Intersection', 'Riemannian' , 'Euclidean', 'Intersection', 'Riemannian' , 'Euclidean', 'Intersection'};
    LabelsML2 = {'Oracle','Oracle','Oracle', 'Oracle' , 'Oracle', 'Oracle', 'Oracle' , 'Oracle', 'Oracle'};
    LabelsML3 = {'-6[dB]','-6[dB]', '-6[dB]', '-10[dB]','-10[dB]', '-10[dB]','-20[dB]','-20[dB]' ,'-20[dB]'};
    LabelsAllSS = {LabelsML1,LabelsML2,LabelsML3};
    % 
    %---------------------------------------------------------------------------------------
    SirMeanSSAll = [SirMeanMusicRMScen1SNR50(:) SirMeanMusicEMScen1SNR50(:) SirMeanIntersectMScen1SNR50(:)....
        SirMeanMusicRMScen12SNR50(:) SirMeanMusicEMScen12SNR50(:)    SirMeanIntersectMScen12SNR50(:) ...
        SirMeanMusicRMScen13SNR50(:) SirMeanMusicEMScen13SNR50(:)    SirMeanIntersectMScen13SNR50(:)];
    LabelsML1 = {'Riemannian','Euclidean','Intersection', 'Riemannian' , 'Euclidean', 'Intersection', 'Riemannian' , 'Euclidean', 'Intersection'};
    LabelsML2 = {'-6[dB]','-6[dB]', '-6[dB]', '-10[dB]','-10[dB]', '-10[dB]','-20[dB]','-20[dB]' ,'-20[dB]'};
    LabelsAllSS = {LabelsML1,LabelsML2};
    MyBoxSubPlotFunc(SirMeanSSAll, LabelsAllSS, 'SIR - SbSp ' ,3);
    saveas(gcf,fullfile('/results', 'Fig4Top_SirSbSp.jpg'))    
    %-----------------------------------------------------------------------------------------
    SirMeanSSORacleAll = [SirMeanMusicROracleMScen1SNR50(:) SirMeanMusicEOracleMScen1SNR50(:) SirMeanIntersectOracleMScen1SNR50(:)....
        SirMeanMusicROracleMScen12SNR50(:) SirMeanMusicEOracleMScen12SNR50(:)    SirMeanIntersectOracleMScen12SNR50(:) ...
        SirMeanMusicROracleMScen13SNR50(:) SirMeanMusicEOracleMScen13SNR50(:)    SirMeanIntersectOracleMScen13SNR50(:)];
    LabelsML1 = {'Riemannian','Euclidean','Intersection', 'Riemannian' , 'Euclidean', 'Intersection', 'Riemannian' , 'Euclidean', 'Intersection'};
    LabelsML2 = {'Oracle','Oracle','Oracle', 'Oracle' , 'Oracle', 'Oracle', 'Oracle' , 'Oracle', 'Oracle'};
    LabelsML3 = {'-6[dB]','-6[dB]', '-6[dB]', '-10[dB]','-10[dB]', '-10[dB]','-20[dB]','-20[dB]' ,'-20[dB]'};
    LabelsAllSS = {LabelsML1,LabelsML2,LabelsML3};
    MyBoxSubPlotFunc(SirMeanSSORacleAll, LabelsAllSS, 'SIR - SbSp Oracle' ,3);
    saveas(gcf,fullfile('/results', 'Fig4Bottom_SirSbSpOracle.jpg'))    
    %------------------------------------------------------------------------------------------------------------------
    DirectivityMvdrAll = [DirectivityMvdrRSaveMScen1SNR50(:) DirectivityMvdrESaveMScen1SNR50(:) DirectivityMvdrRSaveMScen12SNR50(:) DirectivityMvdrESaveMScen12SNR50(:) ];
    LabelsMvdr1 = {'Riemannian','Euclidean', 'Riemannian' , 'Euclidean', 'Riemannian' , 'Euclidean'};
    LabelsMvdr11 = {'Riemannian','Euclidean'};%, 'Riemannian' , 'Euclidean'};%, 'Riemannian' , 'Euclidean'};
    LabelsMvdr2 = {'-6[dB]','-6[dB]', '-10[dB]','-10[dB]','-20[dB]','-20[dB]' };
    LabelsAllMvdr  = {LabelsMvdr1,LabelsMvdr2};
    % 
    
    SirMvdrAll = [SirMeanMvdrRMScen1SNR50(:) SirMeanMvdrEMScen1SNR50(:) SirMeanMvdrRMScen12SNR50(:) SirMeanMvdrEMScen12SNR50(:) SirMeanMvdrRMScen13SNR50(:) SirMeanMvdrEMScen13SNR50(:)];
    LabelsMvdr1 = {'Riemannian','Euclidean', 'Riemannian' , 'Euclidean', 'Riemannian' , 'Euclidean'};
    LabelsMvdr11 = {'Riemannian','Euclidean'};%, 'Riemannian' , 'Euclidean'};%, 'Riemannian' , 'Euclidean'};
    LabelsMvdr2 = {'-6[dB]','-6[dB]', '-10[dB]','-10[dB]','-20[dB]','-20[dB]' };
    LabelsAllMvdr  = {LabelsMvdr1,LabelsMvdr2};
    MyBoxSubPlotFunc(SirMvdrAll, LabelsAllMvdr, 'SIR - MVDR' ,3,[2 2 2]);
    saveas(gcf,fullfile('/results', 'Fig9_MVDR.jpg'))    
end
if PlotFigs_7
    %-------------------------------------------------------------------------------------------------------------
    SirMeanAll = [SirMeanMLRMScen27SNR50(:) SirMeanMLEMScen27SNR50(:) SirMeanMusicRMScen27SNR50(:) SirMeanMusicEMScen27SNR50(:) SirMeanIntersectMScen27SNR50(:)];
    LabelsML1 = {'Riemannian','Euclidean','Riemannian' , 'Euclidean', 'Intersection'};
    LabelsML2 = {'DS','DS','SbSp' , 'SbSp', 'SbSp'};
    LabelsAllML = {LabelsML1,LabelsML2};
    DirectivityAll = [DirectivityMLRSaveMScen27SNR50(:) DirectivityMLESaveMScen27SNR50(:) DirectivityMusicRSaveMScen27SNR50(:) DirectivityMusicESaveMScen27SNR50(:) DirectivityIntersectSaveMScen27SNR50(:)];
    
    MyBoxSubPlotFunc(SirMeanAll, LabelsAllML, 'SIR - Multiple interferences ' ,2,[2,3]);
    saveas(gcf,fullfile('/results', 'Fig7Right_Sir_14Interferences.jpg'))
    % 
end
%---------------------------------------------------------------------------------------------------------------
if exist('SirMeanMLRMScen29SNR50','var')
    SirMeanAll = [SirMeanMLRMScen29SNR50(:) SirMeanMLEMScen29SNR50(:) SirMeanMusicRMScen29SNR50(:) SirMeanMusicEMScen29SNR50(:) SirMeanIntersectMScen29SNR50(:)];
    LabelsML1 = {'Riemannian','Euclidean','Riemannian' , 'Euclidean', 'Intersection'};
    LabelsML2 = {'DS','DS','SbSp' , 'SbSp', 'SbSp'};
    LabelsAllML = {LabelsML1,LabelsML2};
    DirectivityAll = [DirectivityMLRSaveMScen29SNR50(:) DirectivityMLESaveMScen29SNR50(:) DirectivityMusicRSaveMScen29SNR50(:) DirectivityMusicESaveMScen29SNR50(:) DirectivityIntersectSaveMScen29SNR50(:)];
    
    MyBoxSubPlotFunc(SirMeanAll, LabelsAllML, 'SIR - Multiple interferences ' ,2,[2,3]);
    
    MyBoxSubPlotFunc(DirectivityAll, LabelsAllML, 'Directivity - Multiple interferences' ,2,[2 3]);
end
%---------------------------------------------------------------------------------------------------------------
if PlotFigs_8
    SirMeanAll = [SirMeanMLRMScen20SNR50(:) SirMeanMLEMScen20SNR50(:) SirMeanMusicRMScen20SNR50(:) SirMeanMusicEMScen20SNR50(:) SirMeanIntersectMScen20SNR50(:) ];
    DirectivityAll = [DirectivityMLRSaveMScen20SNR50(:) DirectivityMLESaveMScen20SNR50(:) DirectivityMusicRSaveMScen20SNR50(:) DirectivityMusicESaveMScen20SNR50(:) DirectivityIntersectSaveMScen20SNR50(:)];
    LabelsML1 = {'Riemannian','Euclidean','Riemannian' , 'Euclidean', 'Intersection'};
    LabelsML2 = {'DS','DS','SbSp' , 'SbSp', 'SbSp'};
    LabelsAllML = {LabelsML1,LabelsML2};
    MyBoxSubPlotFunc(SirMeanAll, LabelsAllML, 'SIR - Two targets' ,2,[2 3]);
    saveas(gcf,fullfile('/results', 'Fig8Right_Sir_2desired.jpg'))   
    %------------------------------------------------------------------------------------------------------------------
    SirMeanAll = [SirMeanMLRMScen21SNR50(:) SirMeanMLEMScen21SNR50(:) SirMeanMusicRMScen21SNR50(:) SirMeanMusicEMScen21SNR50(:) SirMeanIntersectMScen21SNR50(:) ];
    DirectivityAll = [DirectivityMLRSaveMScen21SNR50(:) DirectivityMLESaveMScen21SNR50(:) DirectivityMusicRSaveMScen21SNR50(:) DirectivityMusicESaveMScen21SNR50(:) DirectivityIntersectSaveMScen21SNR50(:)];
    MyBoxSubPlotFunc(SirMeanAll, LabelsAllML, 'SIR - Two targets' ,2,[2 3]);
    saveas(gcf,fullfile('/results', 'Fig8Left_Sir_2desired.jpg'))
    % 
end

%---------------------------------------------------------------------------------------------------------------
%