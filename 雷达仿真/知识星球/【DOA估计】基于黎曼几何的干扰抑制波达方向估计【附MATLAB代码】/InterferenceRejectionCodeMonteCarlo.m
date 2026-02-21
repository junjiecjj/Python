%%
%%

% 
addpath(fullfile(pwd,'RIR','RIR-Generator-master'));
addpath(fullfile(pwd,'STFT'));
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
SourcesDegApart         = 30; % 
%
KFramesPerInterference  = 0;
ActiveProbaility        = 0.30;                                                                
KAntennas               = 12;                                                               
MidMic                  = ceil(KAntennas/2);                                                 
M                       = KAntennas;
KFramesForCovEst        = 16;                                                                
KFramesForCovEstWithOverlap = KFramesForCovEst*2-1;
KLargeFrames            =  KInterferenceSources;%                                           
BasicSize               = 1024;
KSigSamples             = KLargeFrames*(KFramesForCovEst)*BasicSize;                        
TolDeg                  = 0.1; %                                                                   
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
Lambda                 	= c/FreqBinInter;
%-----------------Acoustic params-------------------------
% Sample frequency (samples/s) 16000
DistBetweenMics         = Lambda/2;%                                                           
MicPosMat               = 1*[1 0 0 ]+[1 1 2] + [DistBetweenMics*(1:KAntennas)' zeros(KAntennas,2)];
TarPos                  = [2 3 1.8];                                                        % 
TarPos2                 = [2 3.5 2.5];
InterPos                = TarPos + [ -1.5 +0.2 -0.5 ; 2.2 0.5 0.1];                         % 
%
if KInterferenceSources > 2
    InterPos                = [4.8 (3.8-1.5) 5.5].* rand(KInterferenceSources,3) + [0 1.5 0]; %
    ThetaInterPosRandDegVec = ThetaLeftDeg + (ThetaRightDeg - ThetaLeftDeg)*rand(KInterferenceSources,1);
end
RoomDims                       = [5 4 3.5];                                                          %  

if IsMCforBeta
    BetaVec = [0.15 0.2 0.25 0.3 0.35 0.4];
    LoopVec = BetaVec;
    % Epsilon is define earlier
else % MC for the noise, epsilon
    beta                    = 0.15 ;%                                                              % 
    EpsilonVec = fliplr(10.^(-([20 30 40 50])/20));
    LoopVec = EpsilonVec;
end% 

for b=1:length(LoopVec)%
    rng(Seed+1,'twister'); % 
    if IsMCforBeta
        beta    = BetaVec(b);
    else
        Epsilon = EpsilonVec(b);
    end   
  
    mtype                   = 'omnidirectional';                                                % Type of microphone
    order                   = -1;%                                                              % -1 equals maximum reflection order!
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
    %     
    %--------------- srcPHAT params -----------------------------
    lsb = TarPos - [1 1 0];
    usb = TarPos + [1 1 0];
    
    %% Plot Params
    linewd                  = 0.8;
    hcfontsize              = 20;
    MarkerSize              = 9;
    %% Flags
    IsPlotPolarFlag         = 0; % 
    %% Funcs
    AngBetweenVecs          = @(u,v) ( acos(real(u/norm(u)*v'/norm(v)))*180/pi ) ;
    DistBetweenVecsFunc     = @(v,u) ( norm(v/norm(v)*(exp(-1j*angle(v(1)))*exp(1j*angle(u(1)))) -u/norm(u)) );
    CorrFunc                = @(u,v) ( abs( u(:)'*v(:)/norm(u)/norm(v) ) );
    DirectivityCalcFunc     = @(Spectrum,Theta_d,DeltaTheta) ( mean(Spectrum(Theta_d)) / (0.5*sum(Spectrum*DeltaTheta)) );
    %% Computing directions to the array - this only make sense if the array is linear and the microphones are close
    ArrayVec                = MicPosMat(2,:) - MicPosMat(1,:);
    % 
    MidArrayPos = MicPosMat(1,:) + ( MicPosMat(end,:) - MicPosMat(1,:) ) / 2;
    ThetaTarDeg             = AngBetweenVecs(TarPos        - MidArrayPos,ArrayVec); 
    ThetaTarDeg2            = AngBetweenVecs(TarPos2        - MidArrayPos,ArrayVec);
    
    %% Interferencs on a circle
    ThetaInterPosRandDegVec = ThetaLeftDeg + (ThetaRightDeg - ThetaLeftDeg)*rand(KInterferenceSources,1);
    ThetaInterPosRandRadVec = ThetaInterPosRandDegVec*pi/180;
    InterPos                = repmat([MidArrayPos(1) MidArrayPos(2) 0],KInterferenceSources,1) + [RSources*cos(ThetaInterPosRandRadVec) RSources*sin(ThetaInterPosRandRadVec) 0.5+2.5*rand(KInterferenceSources,1)];
    ThetaTarInterVec        = ThetaInterPosRandDegVec'; % 
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
    
    %% Creating indicator    
    if KLargeFrames == 2
        IndicatorSmall = eye(KLargeFrames);
    else
        IndicatorSmall = zeros(KLargeFrames,KInterferenceSources);
        IndicatorSmallIndx  = rand(KLargeFrames,KInterferenceSources);
        IndicatorSmall(IndicatorSmallIndx<ActiveProbaility)  = 1;
        IndicatorSmall(IndicatorSmallIndx>=ActiveProbaility) = 0;        
    end    
    IndicatorBig    = kron(IndicatorSmall,ones(KFramesForCovEst*BasicSize,1));
    
    %% Signals @ the Array
    TarPosX = linspace(0.1,RoomDims(1)-0.1,KTarPos);
    TarPosX = linspace(0.3,RoomDims(1)-0.3,KTarPos);
    
    for tp =1:KTarPos
        tp;
        if KTarPos >= 1 % 
            ThetaTarPosRandDeg      = ThetaLeftDeg + (ThetaRightDeg - ThetaLeftDeg)/KTarPos *tp ;%
            ThetaTarPosRandRadVec   = ThetaTarPosRandDeg*pi/180;
            TarPos                  = [MidArrayPos(1) MidArrayPos(2) 0] + [RSources*cos(ThetaTarPosRandRadVec) RSources*sin(ThetaTarPosRandRadVec) TarPos(3)];
        end
        ThetaTarDegVec(tp)          = AngBetweenVecs(TarPos        - MicPosMat(MidMic,:),ArrayVec); % 
        ThetaTarDegVec(tp)          = ThetaTarPosRandDeg; % 
        
        hTarToArrMat                = rir_generator(c, fs, MicPosMat, TarPos, RoomDims, beta, KFiltSamples, mtype, order, dim, orientation, hp_filter);                
        TarSigZeroPad               = [TarSig ; zeros(KFiltSamples,1)];
        TarSigZeroPad2              = [TarSig2 ; zeros(KFiltSamples,1)];
        InterSigZeroPad             = [InterSig ; zeros(KFiltSamples,KInterferenceSources)];    
        
        for MicInd = 1:KAntennas
            
            SigTarAtArray         	= filter(hTarToArrMat(MicInd,:)     ,1,TarSigZeroPad(:,1));            
            SigTarAtArray2        	= filter(hTarToArrMat2(MicInd,:)     ,1,TarSigZeroPad2(:,1));
            
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
        Gamma_R        	= RiemannianMean(GammaTensor);
        Gamma_E         = mean(GammaTensor,3);
        %---------------------Computing Gamma_P - the intersection method------
        Gamma_POracle   = 1;
        Gamma_P         = 1;
        for g = 1: size(GammaTensor,3)
            [EigvecMatG, EigValVecG]    = SortedEVD(GammaTensor(:,:,g));
            U1Oracle                 	= EigvecMatG(:,1:sum(IndicatorSmall(g,:))+1);
            KSources                    = KSourcesFromEigValsFunc(EigValVecG);
            U1                          = EigvecMatG(:,1:KSources);            
            %---------------------------------------------------------------            
            ProjMatOracle               = U1Oracle*(U1Oracle'*U1Oracle)^-1*U1Oracle';
            ProjMat                     = U1*(U1'*U1)^-1*U1';
            Gamma_POracle             	= Gamma_POracle*ProjMatOracle;
            Gamma_P                     = Gamma_P*ProjMat;
            Gamma_P2                    = ProjMat*Gamma_POracle*ProjMat;
        end
        [EigvecMatIntersecOracle, EigValVecIntersecOracle]  = SortedEVD(Gamma_POracle);
        [EigvecMatIntersec, EigValVecIntersec]              = SortedEVD(Gamma_P);
        EigVecIntersectOracle                               = EigvecMatIntersecOracle(:,1:KIntersectionFinalDim);
        KTargetSources                                      = KSourcesFromEigValsFinalFunc(EigValVecIntersec);
        EigVecIntersect                                     = EigvecMatIntersec(:,1:KTargetSources);        
        %----------------------------Spectrum Computation-------------
        %-------For MUSIC-------
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
            StiringVecSpectrum              = exp(2j*pi*Array'/(Lambda)*DistBetweenMics*cos(ThetaVec(i)/180*pi));
            % ---------------------------ML--------------------------------------
            MLSpectrumOfGamma1(i)           = StiringVecSpectrum'*Gamma1*StiringVecSpectrum;
            MLSpectrumOfGamma2(i)           = StiringVecSpectrum'*Gamma2*StiringVecSpectrum;
            MLSpectrumOfGamma_R(i)          = StiringVecSpectrum'*Gamma_R*StiringVecSpectrum;
            MLSpectrumOfGamma_E(i)          = StiringVecSpectrum'*Gamma_E*StiringVecSpectrum;
            % --------------------------MVDR--------------------------------------
            MvdrSpectrumOfGamma1(i)         = 1 / (StiringVecSpectrum'*Gamma1^-1*StiringVecSpectrum);
            MvdrSpectrumOfGamma2(i)         = 1 / (StiringVecSpectrum'*Gamma2^-1*StiringVecSpectrum);
            MvdrSpectrumOfGamma_R(i)        = 1 / (StiringVecSpectrum'*Gamma_R^-1*StiringVecSpectrum);
            MvdrSpectrumOfGamma_E(i)        = 1 / (StiringVecSpectrum'*Gamma_E^-1*StiringVecSpectrum);
            %--------------------------Intersection spectrum-------------------
            IntersectSpectrumOracle(i)     	= norm(StiringVecSpectrum'*EigVecIntersectOracle)^2;
            IntersectSpectrum(i)            = norm(StiringVecSpectrum'*EigVecIntersect)^2;
            %--------------------------MUSIC spectrum--------------------------
            MusicSpectrumOfGamma_R2(i)     	= 1 / (norm(StiringVecSpectrum'*UNoiseGamma_ROracle)^2); % 
            MusicSpectrumOfGamma_ROracle(i)	= abs(((StiringVecSpectrum'*UNoiseGamma_ROracle)*(UNoiseGamma_ROracle'*StiringVecSpectrum)));
            MusicSpectrumOfGamma_R(i)      	= abs(((StiringVecSpectrum'*UNoiseGamma_R)*(UNoiseGamma_R'*StiringVecSpectrum)));
            MusicSpectrumOfGamma_EOracle(i) = (norm(StiringVecSpectrum'*UNoiseGamma_EOracle)^2);
            MusicSpectrumOfGamma_E(i)       = (norm(StiringVecSpectrum'*UNoiseGamma_E)^2);
        end
        
        %%
        for it = 1:KInterferenceSources
            %---------------------------------
            if TargetGain2 > 0 % the 2nd desired source is active 
                SinrMlE(tp,it,b)                = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_E));
                SinrMlR(tp,it,b)                = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_R));
                SinrMvdrE(tp,it)                = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_E));
                SinrMvdrR(tp,it)                = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_R));
                SinrIntersectOracle(tp,it)      = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrumOracle));
                SinrIntersect(tp,it)            = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrum));
                SinrMusicROracle(tp,it)         = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_ROracle));
                SinrMusicEOracle(tp,it)         = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_EOracle));
                SinrMusicR(tp,it)               = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_R));
                SinrMusicE(tp,it)               = SinrCalcFunc(ThetaVec,[ThetaTarDegVec(tp) ThetaTarDeg2],ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_E));
            else
                SinrMlE(tp,it,b)                = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_E));
                SinrMlR(tp,it,b)                = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MLSpectrumOfGamma_R));
                SinrMvdrE(tp,it)                = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_E));
                SinrMvdrR(tp,it)                = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MvdrSpectrumOfGamma_R));
                SinrIntersectOracle(tp,it)      = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrumOracle));
                SinrIntersect(tp,it)            = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(IntersectSpectrum));
                SinrMusicROracle(tp,it)         = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_ROracle));
                SinrMusicEOracle(tp,it)         = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_EOracle));
                SinrMusicR(tp,it)               = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_R));
                SinrMusicE(tp,it)               = SinrCalcFunc(ThetaVec,ThetaTarDegVec(tp),ThetaTarInterVec(it),TolDeg,abs(MusicSpectrumOfGamma_E));
            end
        end % it
        %% Directivity computation
        DeltaThetaRad                   = (ThetaVec(2)-ThetaVec(1))*pi/180;  %[rad]
        DeltaThetaDeg                   = (ThetaVec(2)-ThetaVec(1));         %[deg]
        ThetaTarDegInd                  = find(abs(ThetaVec-ThetaTarDegVec(tp))<0.5*DeltaThetaDeg); % extracting the index in ThetaVec for the target
        if TargetGain2 ~= 0
            ThetaTarDegInd2            	= find(abs(ThetaVec-ThetaTarDeg2)<0.5*DeltaThetaDeg);
            ThetaTarDegInd              = [ ThetaTarDegInd ThetaTarDegInd2];
        end
        DirectivityMLE(tp)              = DirectivityCalcFunc(real(MLSpectrumOfGamma_E),ThetaTarDegInd,DeltaThetaRad);
        DirectivityMLR(tp)              = DirectivityCalcFunc(real(MLSpectrumOfGamma_R),ThetaTarDegInd,DeltaThetaRad);
        DirectivityMusicEOracle(tp)    	= DirectivityCalcFunc(real(MusicSpectrumOfGamma_EOracle),ThetaTarDegInd,DeltaThetaRad);
        DirectivityMusicROracle(tp)    	= DirectivityCalcFunc(real(MusicSpectrumOfGamma_ROracle),ThetaTarDegInd,DeltaThetaRad);
        DirectivityMusicE(tp)           = DirectivityCalcFunc(real(MusicSpectrumOfGamma_E),ThetaTarDegInd,DeltaThetaRad);
        DirectivityMusicR(tp)           = DirectivityCalcFunc(real(MusicSpectrumOfGamma_R),ThetaTarDegInd,DeltaThetaRad);
        DirectivityIntersectOracle(tp)	= DirectivityCalcFunc(real(IntersectSpectrumOracle),ThetaTarDegInd,DeltaThetaRad);
        DirectivityIntersect(tp)        = DirectivityCalcFunc(real(IntersectSpectrum),ThetaTarDegInd,DeltaThetaRad);        

        if IsPlotPolarFlag            
            PolarSpectrumPlot;
        end
    end % tp
end % b Monte Carlo

%%---------------------------------------------------
function Sinr = SinrCalcFunc(ThetaVec,ThetaTarDeg,ThetaTarInter,TolDeg,Spectrum)
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

