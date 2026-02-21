
%%
% 
%%
PlotMusicFlag = 0;
%%
ThetaTarDeg             = AngBetweenVecs(TarPos        - MidArrayPos,ArrayVec);

rLimScale = [-20 0 ];%[-28 0 ];
rLimScale2 = [-20 0 ];

ThetaVecRadFull = [-fliplr(ThetaVec(1:end-1)) ThetaVec]/180*pi;
MLSpectrumOfGamma1Full = DuplicateSpectrumFunc(MLSpectrumOfGamma1);
MLSpectrumOfGamma2Full = DuplicateSpectrumFunc(MLSpectrumOfGamma2);
MLSpectrumOfGamma_EFull = DuplicateSpectrumFunc(MLSpectrumOfGamma_E);
MLSpectrumOfGamma_RFull = DuplicateSpectrumFunc(MLSpectrumOfGamma_R);
IntersectSpectrumFull = DuplicateSpectrumFunc(IntersectSpectrum);
if exist('MusicSpectrumOfGamma_E','var') && PlotMusicFlag
    MusicSpectrumOfGamma_EFull = DuplicateSpectrumFunc(MusicSpectrumOfGamma_E);
    MusicSpectrumOfGamma_RFull = DuplicateSpectrumFunc(MusicSpectrumOfGamma_R);
end

ThetaVecRadFull = ThetaVec/180*pi;
%  
figure;
subplot(2,1,1);
polarplot(ThetaVecRadFull,MLSpectrumOfGamma_RFull,'LineWidth',3); hold on
% 
s = polarplot(ThetaVecRadFull,MLSpectrumOfGamma_EFull,':','LineWidth',3); 
s.Color(4)=0.9;
p = polarplot(ThetaTarDeg/180*pi*ones(1,2),rLimScale2,'LineWidth',2);
p.Color = 'black';
if exist('ThetaTarDeg2','var') && PlotMusicFlag
    if TargetGain2~=0
        p = polarplot(ThetaTarDeg2/180*pi*ones(1,2),rLimScale,'LineWidth',2);
        p.Color = 'black';
    end
end
if KInterferenceSources < 10
    for it=1:KInterferenceSources
        p = polarplot(ThetaTarInterVec(it)/180*pi*ones(1,2),rLimScale2,'-.','LineWidth',2);
        p.Color = 'black';
    end
end
rlim(rLimScale2);
% 
legend('\Gamma_R (Riem)','\Gamma_E (Euc)','Desired','Interference');
thetalim([0 180])

subplot(2,1,2);
if ~exist('MusicSpectrumOfGamma_E','var') || ~PlotMusicFlag
    polarplot(ThetaVecRadFull,MLSpectrumOfGamma1Full,'LineWidth',3,'Color',OrangeColDef); hold on%OrangeColDef
    s = polarplot(ThetaVecRadFull,MLSpectrumOfGamma2Full,'LineWidth',3,'Color',[0.9 0.49 0]);%
else
    polarplot(ThetaVecRadFull,MusicSpectrumOfGamma_RFull,'LineWidth',3,'Color',OrangeColDef); hold on
    s = polarplot(ThetaVecRadFull,MusicSpectrumOfGamma_EFull,':','LineWidth',3,'Color',[0.9 0.49 0]);
end
% 
p = polarplot(ThetaTarDeg/180*pi*ones(1,2),rLimScale,'LineWidth',2);
p.Color = 'black';
if exist('ThetaTarDeg2','var')
    if TargetGain2~=0
        p = polarplot(ThetaTarDeg2/180*pi*ones(1,2),rLimScale,'LineWidth',2);
        p.Color = 'black';
    end
end
if KInterferenceSources < 10
    for it=1:KInterferenceSources
        p = polarplot(ThetaTarInterVec(it)/180*pi*ones(1,2),rLimScale,'-.','LineWidth',2);
        p.Color = 'black';
    end
end
rlim(rLimScale);
if ~exist('MusicSpectrumOfGamma_E','var') || ~PlotMusicFlag
    legend('\Gamma_1 (1^{st} segment)','\Gamma_2 (2^{nd} segment','Desired','Interference');
else
    legend('\Gamma_R (MUSIC)','\Gamma_E (MUSIC)','Intersect','Desired','Interference');
end
thetalim([0 180])
% 
%  
saveas(gcf,fullfile('/results', 'Fig2_BeamPattern.jpg'))


%%
function SpectrumFull = DuplicateSpectrumFunc(Spectrum)
    SpectrumFull = 10*log10(abs(Spectrum)/max(abs(Spectrum)));
end

