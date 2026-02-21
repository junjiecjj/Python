%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% !!! Please note that the main file is LoopWrapper.m !!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
addpath(fullfile(pwd,'RIR','RIR-Generator-master'));
addpath(fullfile(pwd,'STFT'));
%%
%% Colors definition
ColorBlueMat    = [ 0    0.4510    0.7412 ;  0.0980    0.5882    0.8902 ; 0.3020    0.7451    0.9604 ;  0.5137    0.8235    0.9882];
ColorRedMat     = [0.6196    0.0627    0.1647 ; 0.7882    0.1098    0.2353 ; 0.9216    0.2118    0.3412 ; 0.9608    0.4118    0.5137];
OrangeColDef    = [0.9290    0.6940    0.1250];
RedColDef       = [0.8500    0.3250    0.0980];
%% Params
c                       = 340;  % Sound velocity (m/s)
RSources                = 2; %[m] 
ThetaLeftDeg            = 20;
ThetaRightDeg           = 160;
%
KFramesPerInterference  = 0;
ActiveProbaility        = 0.30;                                                              
KAntennas               = 12;                                                                 
MidMic                  = ceil(KAntennas/2);                                                 
M                       = KAntennas;
KFramesForCovEst        = 16;                                                                
KFramesForCovEstWithOverlap = KFramesForCovEst*2-1;
BasicSize               = 1024;
KSigSamples             = KLargeFrames*(KFramesForCovEst)*BasicSize;                        
TolDeg                  = 0.1;                                                                
KTarPos                 = 2e1;                                                         	  
%-----------Intersection Params
KIntersectionFinalDim   = 1; % 
%----------------------MUSIC----------------------------
KSignals4MUSIC          = 1;
% ----------Interference amplitude gain---------------
TargetGain              = 1;
InputSir                = 20*log10(TargetGain /mean(InterferenceAmpGain));
% ----------------noise params------------------------
Snr_dB                  = 50;  
Epsilon                 = 1/10^(Snr_dB / 20);
%----------------Acoustic Filter length----------------
KFiltSamples            = 2*BasicSize;
%-----------------STFT Params-------------------------
WinLength               = BasicSize;% 
FactBetweenFiltLengths  = KFiltSamples/WinLength;
%-------------Spectrum computation params---------------
KBins                   = 1;                                                                    
Binvec                  = 500/FactBetweenFiltLengths;  
BinInd                  = Binvec(1); % 
%------------------ Frequency Params------------------
fs                      = 16000;
FreqBinInter          	= fs/WinLength*(BinInd-0.5); %  
Lambda                  = c/FreqBinInter;
%-----------------Acoustic params-------------------------
% Sample frequency (samples/s) 16000
DistBetweenMics         = Lambda/2;%                                                             
MicPosMat               = 1*[1 0 0 ]+[1 1 2] + [DistBetweenMics*(1:KAntennas)' zeros(KAntennas,2)];     
TarPos                  = [2 3 1.8];                                                          
TarPos2                 = [2 3.5 2.5];
%  
RoomDims                = [5 4 3.5];  %
beta                    = 0.15 ;%

mtype                   = 'omnidirectional';                                                % Type of microphone
order                   = -1;%-1;                                                           % -1 equals maximum reflection order!
dim                     = 3;                                                                % Room dimension
orientation             = 0;                                                                % Microphone orientation (rad)
hp_filter               = 1;                                                                % Enable high-pass filter
%-----------------Fft params--------------------------
Nfft                    = KFiltSamples;%
Nfft2                   = Nfft/2;
f                       = [0:Nfft2 -Nfft2+1:-1]'/Nfft;
df                      = f(2)-f(1);
%---
Array                   = 0:M-1; 
%--------------- srcPHAT params -----------------------------
lsb = TarPos - [1 1 0];
usb = TarPos + [1 1 0];

%% Plot Params
linewd                  = 0.8;
hcfontsize              = 20;
MarkerSize              = 9;
%% Flags
IsPlotPolarFlag         = 1;   
%% Funcs
AngBetweenVecs          = @(u,v) ( acos(real(u/norm(u)*v'/norm(v)))*180/pi ) ;
DistBetweenVecsFunc     = @(v,u) ( norm(v/norm(v)*(exp(-1j*angle(v(1)))*exp(1j*angle(u(1)))) -u/norm(u)) );
CorrFunc                = @(u,v) ( abs( u(:)'*v(:)/norm(u)/norm(v) ) );
DirectivityCalcFunc     = @(Spectrum,Theta_d,DeltaTheta) ( mean(Spectrum(Theta_d)) / (0.5*sum(Spectrum*DeltaTheta)) );

%% 
ArrayVec                = MicPosMat(2,:) - MicPosMat(1,:);

MidArrayPos = MicPosMat(1,:) + ( MicPosMat(end,:) - MicPosMat(1,:) ) / 2;
ThetaTarDeg             = AngBetweenVecs(TarPos        - MidArrayPos,ArrayVec);   
ThetaTarDeg2            = AngBetweenVecs(TarPos2        - MidArrayPos,ArrayVec);   
%%
if KInterferenceSources > 2    
    InterPos                = [4.8 (3.8-1.5) 5.5].* rand(KInterferenceSources,3) + [0 1.5 0]; % Considering Room dimensions and 'MicPosMat' location (we want to be on one side)
    ThetaInterPosRandDegVec = ThetaLeftDeg + (ThetaRightDeg - ThetaLeftDeg)*rand(KInterferenceSources,1);
end
%% Interferences on a circle
ThetaInterPosRandDegVec = ThetaLeftDeg + (ThetaRightDeg - ThetaLeftDeg)*rand(KInterferenceSources,1);

ThetaInterPosRandRadVec = ThetaInterPosRandDegVec*pi/180;
InterPos                = repmat([MidArrayPos(1) MidArrayPos(2) 0],KInterferenceSources,1) + [RSources*cos(ThetaInterPosRandRadVec) RSources*sin(ThetaInterPosRandRadVec) 0.5+2.5*rand(KInterferenceSources,1)];
ThetaTarInterVec        = ThetaInterPosRandDegVec';   
%% RIR creation - generating RIR
hTarToArrMat2           = rir_generator(c, fs, MicPosMat, TarPos2, RoomDims, beta, KFiltSamples, mtype, order, dim, orientation, hp_filter);

for it = 1:KInterferenceSources
    hInterToArrTens(it,:,:) = rir_generator(c, fs, MicPosMat, InterPos(it,:), RoomDims, beta, KFiltSamples, mtype, order, dim, orientation, hp_filter);
end

%% Creating the signals 
TarSig                  = randn(KSigSamples,1);
NormFactor              = norm(TarSig);
TarSigClean             = TarSig  / NormFactor;
TarSig                  = TargetGain * TarSig / NormFactor;

TarSig2                  = randn(KSigSamples,1);
NormFactor2              = norm(TarSig2);
TarSigClean2             = TarSig2  / NormFactor2;
TarSig2                  = TargetGain2 * TarSig2 / NormFactor2;

InterSigClean           = randn(KSigSamples,KInterferenceSources);%/ NormFactor;
for it=1:KInterferenceSources
    InterSigClean(:,it) = InterSigClean(:,it) / norm(InterSigClean(:,it));
end
InterSig                = InterferenceAmpGain .* InterSigClean;

%% Creating activity indicator

if KLargeFrames == 2
    IndicatorSmall = eye(KLargeFrames);
else
    IndicatorSmall = zeros(KLargeFrames,KInterferenceSources);
    IndicatorSmallIndx  = rand(KLargeFrames,KInterferenceSources);
    IndicatorSmall(IndicatorSmallIndx<ActiveProbaility)  = 1;
    IndicatorSmall(IndicatorSmallIndx>=ActiveProbaility) = 0;
end
% 
IndicatorBig    = kron(IndicatorSmall,ones(KFramesForCovEst*BasicSize,1));
 
%% Signals @ the Array

for tp =1:KTarPos
    tp;
    if KTarPos >= 1 % 
        %          
     	ThetaTarPosRandDeg = ThetaLeftDeg + (ThetaRightDeg - ThetaLeftDeg)/KTarPos *tp ;%
        ThetaTarPosRandRadVec = ThetaTarPosRandDeg*pi/180;
        TarPos                  = [MidArrayPos(1) MidArrayPos(2) 0] + [RSources*cos(ThetaTarPosRandRadVec) RSources*sin(ThetaTarPosRandRadVec) TarPos(3)];
        if tp == 5 % for plotting 
            TarPos4PlotScenario = TarPos;
        end
    end
    ThetaTarDegVec(tp)      = AngBetweenVecs(TarPos        - MicPosMat(MidMic,:),ArrayVec); % 
    ThetaTarDegVec(tp)      = ThetaTarPosRandDeg; % 

    hTarToArrMat            = rir_generator(c, fs, MicPosMat, TarPos, RoomDims, beta, KFiltSamples, mtype, order, dim, orientation, hp_filter);
    
    TarSigZeroPad           = [TarSig ; zeros(KFiltSamples,1)];
    TarSigZeroPad2          = [TarSig2 ; zeros(KFiltSamples,1)];
    InterSigZeroPad         = [InterSig ; zeros(KFiltSamples,KInterferenceSources)];
     
    for MicInd = 1:KAntennas       
        
        SigTarAtArray                   = filter(hTarToArrMat(MicInd,:)     ,1,TarSigZeroPad(:,1));        
        SigTarAtArray2                  = filter(hTarToArrMat2(MicInd,:)     ,1,TarSigZeroPad2(:,1));        
        
        for it=1:KInterferenceSources
            SigInterAtArrayMat(:,it)    = filter(squeeze(hInterToArrTens(it,MicInd,:))  ,1,InterSigZeroPad(:,it));
        end        
        % 
        SigTarAtArray                   = SigTarAtArray(1:KSigSamples);       
 
        SigTarAtArray2                  = SigTarAtArray2(1:KSigSamples);
        SigInterAtArrayMatTrim          = SigInterAtArrayMat(1:KSigSamples,:);
        SigInterAtArrayMatMasked        = SigInterAtArrayMatTrim.*IndicatorBig;
        NoiseVec                        = randn(KSigSamples,1);
        KFactNoise                      = norm(NoiseVec);
        NoiseVec                        = Epsilon*randn(KSigSamples,1) / KFactNoise; %
        
        SigForGammaTimeDomain(MicInd,:) = SigTarAtArray2 + SigTarAtArray + sum(SigInterAtArrayMatMasked,2) + NoiseVec;
        SigForGammaStft                 = stft(SigForGammaTimeDomain(MicInd,:),WinLength);
        SigForGammaStftFreqBin(MicInd,:)          = SigForGammaStft(BinInd,:); 
    end % MicInd
    
    %% Computing Covariance matrices
    GammaTensor = zeros(KAntennas,KAntennas,KLargeFrames);
    for g = 1:KLargeFrames
        SigForGamma     = SigForGammaStftFreqBin(:,(g-1)*KFramesForCovEstWithOverlap+g:g*KFramesForCovEstWithOverlap+g-1);
        for sftfFrame = 1:KFramesForCovEstWithOverlap
            GammaTensor(:,:,g) = GammaTensor(:,:,g) + 1/size(SigForGamma,2)*SigForGamma(:,sftfFrame)*SigForGamma(:,sftfFrame)';
        end
    end    
    Gamma1 = squeeze(GammaTensor(:,:,1));
    Gamma2 = squeeze(GammaTensor(:,:,2));
    %% Computing spectrum
    % 
    ThetaVec     	= linspace(0,180,1e4+1); % 
    Gamma_R      	= RiemannianMean(GammaTensor);
    Gamma_E     	= mean(GammaTensor,3);   
    %---------------------Computing Gamma_P - the intersection method------
    Gamma_POracle = 1;
    Gamma_P = 1;
    for g = 1: size(GammaTensor,3)
        [EigvecMatG, EigValVecG]    = SortedEVD(GammaTensor(:,:,g));
        U1Oracle                          = EigvecMatG(:,1:sum(IndicatorSmall(g,:))+1);
        KSources                    = KSourcesFromEigValsFunc(EigValVecG);
        U1                          = EigvecMatG(:,1:KSources);
        %---------------------------------------------------------------        
        ProjMatOracle               = U1Oracle*(U1Oracle'*U1Oracle)^-1*U1Oracle';
        ProjMat                     = U1*(U1'*U1)^-1*U1';
        Gamma_POracle              	= Gamma_POracle*ProjMatOracle;
        Gamma_P                     = Gamma_P*ProjMat;        
        Gamma_P2                    = ProjMat*Gamma_P*ProjMat;
    end
    [EigvecMatIntersecOracle, EigValVecIntersecOracle]  = SortedEVD(Gamma_POracle);
    [EigvecMatIntersec, EigValVecIntersec]  = SortedEVD(Gamma_P);
    EigVecIntersectOracle = EigvecMatIntersecOracle(:,1:KIntersectionFinalDim);
    KTargetSources = KSourcesFromEigValsFinalFunc(EigValVecIntersec);
    EigVecIntersect = EigvecMatIntersec(:,1:KTargetSources);
   
    %----------------------------Spectrum Computation-------------
    %--------for MUSIC-------
        UNoiseGamma_EOracle         = ExtractSigSubSpaceFunc(Gamma_E,KSignals4MUSIC);
        UNoiseGamma_ROracle         = ExtractSigSubSpaceFunc(Gamma_R,KSignals4MUSIC);
        EigGamma_E                  = real(eig(Gamma_E));
        KTargetSources_E            = KSourcesFromEigValsFinalFunc(EigGamma_E);        
        UNoiseGamma_E               = ExtractSigSubSpaceFunc(Gamma_E,KTargetSources_E);
        EigGamma_R                  = real(eig(Gamma_R));
        KTargetSources_R            = KSourcesFromEigValsFinalFunc(EigGamma_R);        
        UNoiseGamma_R               = ExtractSigSubSpaceFunc(Gamma_R,KTargetSources_R);            
    %----------------
    for i = 1:length(ThetaVec)
        StiringVecSpectrum                  = exp(2j*pi*Array'/(Lambda)*DistBetweenMics*cos(ThetaVec(i)/180*pi));
        % ---------------------------ML--------------------------------------
        MLSpectrumOfGamma1(i)               = StiringVecSpectrum'*Gamma1*StiringVecSpectrum;% 1 / (StiringVecSpectrum'*P1^-1*StiringVecSpectrum);
        MLSpectrumOfGamma2(i)               = StiringVecSpectrum'*Gamma2*StiringVecSpectrum;%1 / (StiringVecSpectrum'*P2^-1*StiringVecSpectrum);
        MLSpectrumOfGamma_R(i)              = StiringVecSpectrum'*Gamma_R*StiringVecSpectrum;%1 / (StiringVecSpectrum'*PMid^-1*StiringVecSpectrum);
        MLSpectrumOfGamma_E(i)              = StiringVecSpectrum'*Gamma_E*StiringVecSpectrum;%1 / (StiringVecSpectrum'*PMid^-1*StiringVecSpectrum);
        % --------------------------MVDR--------------------------------------
        MvdrSpectrumOfGamma1(i)             = 1 / (StiringVecSpectrum'*Gamma1^-1*StiringVecSpectrum);
        MvdrSpectrumOfGamma2(i)             = 1 / (StiringVecSpectrum'*Gamma2^-1*StiringVecSpectrum);
        MvdrSpectrumOfGamma_R(i)            = 1 / (StiringVecSpectrum'*Gamma_R^-1*StiringVecSpectrum);
        MvdrSpectrumOfGamma_E(i)            = 1 / (StiringVecSpectrum'*Gamma_E^-1*StiringVecSpectrum);
        %--------------------------Intersection spectrum-------------------
        IntersectSpectrumOracle(i)         	= norm(StiringVecSpectrum'*EigVecIntersectOracle)^2;
        IntersectSpectrum(i)                = norm(StiringVecSpectrum'*EigVecIntersect)^2;
        %--------------------------MUSIC spectrum--------------------------
        MusicSpectrumOfGamma_R2(i)          = 1 / (norm(StiringVecSpectrum'*UNoiseGamma_ROracle)^2); % 
        MusicSpectrumOfGamma_ROracle(i) 	= abs(((StiringVecSpectrum'*UNoiseGamma_ROracle)*(UNoiseGamma_ROracle'*StiringVecSpectrum)));
        MusicSpectrumOfGamma_R(i)           = abs(((StiringVecSpectrum'*UNoiseGamma_R)*(UNoiseGamma_R'*StiringVecSpectrum)));        
        MusicSpectrumOfGamma_EOracle(i)   	= (norm(StiringVecSpectrum'*UNoiseGamma_EOracle)^2);
        MusicSpectrumOfGamma_E(i)           = (norm(StiringVecSpectrum'*UNoiseGamma_E)^2);
    end
    
    %%
    for it = 1:KInterferenceSources       
        %---------------------------------
        if TargetGain2 > 0 % the 2nd desired source is active 
            SinrMlE(tp,it)          = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_E));
            SinrMlR(tp,it)          = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_R));
            SinrMvdrE(tp,it)        = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_E));
            SinrMvdrR(tp,it)        = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_R));
            SinrIntersectOracle(tp,it)    = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrumOracle));
            SinrIntersect(tp,it)    = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrum));
            SinrMusicROracle(tp,it)       = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_ROracle));
            SinrMusicEOracle(tp,it)       = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_EOracle));
            SinrMusicR(tp,it)       = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_R));
            SinrMusicE(tp,it)       = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_E));
        else
            SinrMlE(tp,it)          = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_E));
            SinrMlR(tp,it)          = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_R));
            SinrMvdrE(tp,it)        = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_E));
            SinrMvdrR(tp,it)        = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_R));
            SinrIntersectOracle(tp,it)    = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrumOracle));
            SinrIntersect(tp,it)    = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrum));
            SinrMusicROracle(tp,it)       = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_ROracle));
            SinrMusicEOracle(tp,it)       = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_EOracle));
            SinrMusicR(tp,it)       = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_R));
            SinrMusicE(tp,it)       = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_E));
        end
    end % it
     
    %% Directivity computation
    DeltaThetaRad                   = (ThetaVec(2)-ThetaVec(1))*pi/180;  %[rad]
    DeltaThetaDeg                   = (ThetaVec(2)-ThetaVec(1));         %[deg]
    ThetaTarDegInd                  = find(abs(ThetaVec-ThetaTarDegVec(tp))<0.5*DeltaThetaDeg); % 
    if TargetGain2 ~= 0
        ThetaTarDegInd2             = find(abs(ThetaVec-ThetaTarDeg2)<0.5*DeltaThetaDeg);
        ThetaTarDegInd = [ ThetaTarDegInd ThetaTarDegInd2];
    end
    DirectivityMLE(tp)              = DirectivityCalcFunc(real(MLSpectrumOfGamma_E),ThetaTarDegInd,DeltaThetaRad);
    DirectivityMLR(tp)              = DirectivityCalcFunc(real(MLSpectrumOfGamma_R),ThetaTarDegInd,DeltaThetaRad);
    DirectivityMvdrE(tp)            = DirectivityCalcFunc(real(MvdrSpectrumOfGamma_E),ThetaTarDegInd,DeltaThetaRad);
    DirectivityMvdrR(tp)            = DirectivityCalcFunc(real(MvdrSpectrumOfGamma_R),ThetaTarDegInd,DeltaThetaRad);
    DirectivityMusicEOracle(tp)   	= DirectivityCalcFunc(real(MusicSpectrumOfGamma_EOracle),ThetaTarDegInd,DeltaThetaRad);
    DirectivityMusicROracle(tp)    	= DirectivityCalcFunc(real(MusicSpectrumOfGamma_ROracle),ThetaTarDegInd,DeltaThetaRad);
    DirectivityMusicE(tp)           = DirectivityCalcFunc(real(MusicSpectrumOfGamma_E),ThetaTarDegInd,DeltaThetaRad);
    DirectivityMusicR(tp)           = DirectivityCalcFunc(real(MusicSpectrumOfGamma_R),ThetaTarDegInd,DeltaThetaRad);
    DirectivityIntersectOracle(tp) 	= DirectivityCalcFunc(real(IntersectSpectrumOracle),ThetaTarDegInd,DeltaThetaRad);
    DirectivityIntersect(tp)        = DirectivityCalcFunc(real(IntersectSpectrum),ThetaTarDegInd,DeltaThetaRad);
    %% Extracting directions 
    ThetaEst_E(tp)                  = DoAFromSpectrumFunc(ThetaVec,MLSpectrumOfGamma_E);
    ThetaEst_R(tp)                  = DoAFromSpectrumFunc(ThetaVec,MLSpectrumOfGamma_R);
    ThetaEst_Intersect(tp)          = DoAFromSpectrumFunc(ThetaVec,IntersectSpectrum);
    %-----src PHAT-----
    SigForSrcPhat                   = transpose(SigForGammaTimeDomain);    
    [finalpos,finalsrp,finalfe]     = srplems(SigForSrcPhat, MicPosMat, fs, lsb, usb);
    ThetaEst_srcPHAT(tp)            = AngBetweenVecs(finalpos        - MicPosMat(MidMic,:),ArrayVec);
    %%
    if IsPlotPolarFlag && Mscen==1 && (tp== 5) && scen==1 %          
        PolarSpectrumPlot;
    end
end % tp
DirectionError_E        = (ThetaEst_E - ThetaTarDegVec);
DirectionError_R        = (ThetaEst_R - ThetaTarDegVec);
DirectionError_Intersect        = (ThetaEst_Intersect - ThetaTarDegVec);
DirectionError_srcPHAT  = (ThetaEst_srcPHAT - ThetaTarDegVec);

%%
if (Mscen == 1 || Mscen == 12) && scen==1
    %% ------------DoA estimation error -------------
    figure;hold on
    h = plot(ThetaTarDegVec,ThetaEst_R,'s','MarkerSize',13);
    set(h, 'MarkerFaceColor', get(h,'Color'));
    h = plot(ThetaTarDegVec,ThetaEst_E,'o','MarkerSize',9);
    set(h, 'MarkerFaceColor', get(h,'Color'));
    h = plot(ThetaTarDegVec,ThetaEst_Intersect,'p','MarkerSize',13);
    set(h, 'MarkerFaceColor', get(h,'Color'));
    %      
    h2 = plot(ThetaTarDegVec,ThetaEst_srcPHAT,'>','MarkerSize',9);
    set(h2, 'MarkerFaceColor', get(h2,'Color'));
    plot(ThetaTarDegVec,ThetaTarDegVec,'k');
    for it=1:KInterferenceSources
        yline(ThetaTarInterVec(it),'--');
    end
    %  
    legend('Riem','Euc','Itersection','SRP-PHAT','Desired','Interferece','Location','Best');%
    xlabel('Target direction [deg]');
    % 
    ylabel('[deg]');
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    box on ; grid on
    saveas(gcf,fullfile('/results', ['Fig5_DoAEstSirMinus' num2str(round(abs(InputSir))) '.jpg']))
end
if Mscen == 27  && scen==1 % plots only for 14 interferences and only once
 %% ---------- Activity map for the interferences---------------
    map = [ 1 1 1; 	0 0.4470 0.7410];
    IndicatorSmallT = IndicatorSmall';
    figure; hold on
    imagesc([1:size(IndicatorSmallT,2)]-0.5, [1:size(IndicatorSmallT,1)]-0.5, IndicatorSmallT);%grid on
    axis xy
    % 
    xlabel('Segment index');
    ylabel('Interference index');
    yt = get(gca, 'YTick');                         
    ytlbl = (1:KInterferenceSources);               
    set(gca, 'YTick',ytlbl-0.5, 'YTickLabel',ytlbl)
    % 
    yt = get(gca, 'XTick');                         
    xtlbl = (1:KLargeFrames);                     % 
    set(gca, 'XTick',xtlbl-0.5, 'XTickLabel',xtlbl)
    % 
    for l=0:KInterferenceSources
        yline(l);
    end
    for l=0:KLargeFrames
        xline(l);
    end
    % colormap gray
    colormap(map);
    alpha(0.7);
    box on
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    saveas(gcf,fullfile('/results', 'Fig7Left_ActivationMap.jpg'))
end   
if Mscen==1 && scen==1
    %% ---------Plot scenario----------------
    figure; hold on
    %
    for m=1:size(MicPosMat,1)
        plot3(MicPosMat(m,1),MicPosMat(m,2),MicPosMat(m,3),'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on
        plot3(TarPos4PlotScenario(1,1),TarPos4PlotScenario(1,2),TarPos4PlotScenario(1,3),'p','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r');
        if m <= size(InterPos,1)
            plot3(InterPos(m,1),InterPos(m,2),InterPos(m,3),'s','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g');
        end
    end
    for m=size(MicPosMat,1)+1:KInterferenceSources % 
        plot3(InterPos(m,1),InterPos(m,2),InterPos(m,3),'s','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g');
    end
    plotcube(RoomDims,[ 0  0  0],0.01,[1 0 0]);
    xlim([0 5])
    xlabel('x[m]');ylabel('y[m]');zlabel('z[m]');
    legend('Microphone','Target','Interference');
    grid on
%     if SaveFigFlag
%         savefig([pwd '\FiguresSave\Scenario3D']);
%     end
     ThetaVecPlot = pi/180*linspace(ThetaLeftDeg,ThetaRightDeg,2e2);
     saveas(gcf,fullfile('/results', 'Fig1Left_Scenario3D.jpg'))
    figure; hold on    
    for m=1:size(MicPosMat,1)
        plot(MicPosMat(m,1),MicPosMat(m,2),'o','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','b'); hold on
        plot(TarPos4PlotScenario(1,1),TarPos4PlotScenario(1,2),'p','MarkerSize',10,'MarkerEdgeColor','r','MarkerFaceColor','r');
        if m <= size(InterPos,1)
            plot(InterPos(m,1),InterPos(m,2),'s','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g');
        end
    end
    for m=size(MicPosMat,1)+1:KInterferenceSources % 
        plot(InterPos(m,1),InterPos(m,2),'s','MarkerSize',10,'MarkerEdgeColor','g','MarkerFaceColor','g');
    end
    plot(MidArrayPos(1) + RSources*cos(ThetaVecPlot), MidArrayPos(2) + RSources*sin(ThetaVecPlot),'k','LineWidth',3)        
    xlim([0 5])
    xlabel('x[m]');ylabel('y[m]');zlabel('z[m]');
    legend('Microphone','Target','Interference');
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    set(gca, 'YTick', 1:4);
    box on ;grid on   
    axis equal 
    saveas(gcf,fullfile('/results', 'Fig1Right_Scenario2D.jpg'))
    
%     if SaveFigFlag
%         savefig([pwd '\FiguresSave\Scenario2D']);
%     end
 end
%%
function plotcube(varargin)
% 
%

% 
inArgs = { ...
    [10 56 100] , ... % 
    [10 10  10] , ... % 
    .7          , ... % 
    [1 0 0]       ... % 
    };
% 
inArgs(1:nargin) = varargin;
% 
[edges,origin,alpha,clr] = deal(inArgs{:});
XYZ = { ...
    [0 0 0 0]  [0 0 1 1]  [0 1 1 0] ; ...
    [1 1 1 1]  [0 0 1 1]  [0 1 1 0] ; ...
    [0 1 1 0]  [0 0 0 0]  [0 0 1 1] ; ...
    [0 1 1 0]  [1 1 1 1]  [0 0 1 1] ; ...
    [0 1 1 0]  [0 0 1 1]  [0 0 0 0] ; ...
    [0 1 1 0]  [0 0 1 1]  [1 1 1 1]   ...
    };
XYZ = mat2cell(...
    cellfun( @(x,y,z) x*y+z , ...
    XYZ , ...
    repmat(mat2cell(edges,1,[1 1 1]),6,1) , ...
    repmat(mat2cell(origin,1,[1 1 1]),6,1) , ...
    'UniformOutput',false), ...
    6,[1 1 1]);
cellfun(@patch,XYZ{1},XYZ{2},XYZ{3},...
    repmat({clr},6,1),...
    repmat({'FaceAlpha'},6,1),...
    repmat({alpha},6,1)...
    );
view(3);
end
%%---------------------------------------------------
function Sinr = SinrCalcFunc(ThetaVec,ThetaTarDeg,ThetaTarInter,TolDeg,Spectrum)
% 
% 
if length(ThetaTarDeg)==1
    IndSpecTar = find(abs(ThetaVec-ThetaTarDeg)<TolDeg);
    IndSpecInter = find(abs(ThetaVec-ThetaTarInter)<TolDeg);
    SpectrumNorm = Spectrum/max(abs(Spectrum));
    Sinr = 10*log10(max(abs(SpectrumNorm(IndSpecTar)))) -  10*log10(max(abs(SpectrumNorm(IndSpecInter))));
else
    for t=1:length(ThetaTarDeg)
        IndSpecTar = find(abs(ThetaVec-ThetaTarDeg(t))<TolDeg);
        IndSpecInter = find(abs(ThetaVec-ThetaTarInter)<TolDeg);
        SpectrumNorm = Spectrum/max(abs(Spectrum));         
        SinrTmp = 10*log10(max(abs(SpectrumNorm(IndSpecTar)))) -  10*log10(max(abs(SpectrumNorm(IndSpecInter))));
        SinrLin(t) = 10^(SinrTmp/10);        
    end
    Sinr = 10*log10(mean(SinrLin));
end

end
%%---------------------------------------------------
function [EigvecMat, EigValVec] = SortedEVD(Mat)

[Phi, LambdaMat]    = eig(Mat);
[~,ind]             = sort(diag(LambdaMat),'desc');
LambdaMat           = LambdaMat(ind,ind);
EigvecMat        	= Phi(:,ind);
EigValVec           = diag(LambdaMat);

end

%%----------------------------------------------
function Ksources = KSourcesFromEigValsFunc(EigValVecG)

EigValVecG  = real(EigValVecG) / sum(real(EigValVecG));
Threshold   = 1.5*mean(EigValVecG);
Ksources    = sum(EigValVecG>Threshold);

end

function Ksources = KSourcesFromEigValsFinalFunc(EigValVecG)

EigValVecG  = real(EigValVecG) / sum(real(EigValVecG));
Threshold   = mean(EigValVecG) + std(EigValVecG);
Ksources    = sum(EigValVecG>Threshold);

end

