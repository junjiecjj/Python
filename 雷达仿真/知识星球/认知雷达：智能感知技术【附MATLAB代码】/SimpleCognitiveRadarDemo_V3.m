%% Cognitive Radar Basic
% Author: Dr Anum Pirkani

% Copyright (c) 2025 Anum Pirkani 
% All rights reserved.
% Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
% modify, and distribute this code (the source files) and its documentation for any purpose, provided that the 
% copyright notice in its entirety appears in all copies of this code, and the source of this code, 
% This should be acknowledged in any publication that reports research using this code.

clear; clc; close all;
%rng(18);

NumRangeBins  = 101; 
NumDoplerBins = 64; 
NumScans      = 64;    

% Initial States
  Car.InitialRange  = 100;  
  Car.DopplerBin    = 40;  
  Car.SNRdB         = 10;     
  Car.Velocity      = -0.80;
  Car.VelocityDrift =  0;  

  Pedestrian.InitialRange = 30;   
  Pedestrian.DopplerBin   = 9;     
  Pedestrian.SNRdB        = 8;     
  Pedestrian.GaitFreq     = 0.22;  
  Pedestrian.SidebandW    = 5;      
  Pedestrian.Velocity     = -0.20;  

% Target shapes (range, Doppler std in Gaussian)
  Car.hr0        = 2.2; 
  Car.hd0        = 2.0;
  Pedestrian.hr0 = 2.0; 
  Pedestrian.hd0 = 1.7;

% Background Noise and Interference
  NoiseMu       = 0.4;    % Mean (linear)
  ClutterScale  = 0.55;   % Rayleigh
  InterfProb    = 0.1;    % Probability of Interference per scan
  InterfPower   = 8;      % Strength of Interference
  InterfWidth   = [4 18]; % Range of Interference widths

% CA-CFAR
  CFAR.TrainR = 6;  
  CFAR.TrainD = 6;  
  CFAR.GuardR = 2; 
  CFAR.GuardD = 2; 
  CFAR.k      = 4.2;          

% Cognitive Adaptation
  Alpha.power   = 0.08;       % Attention Power Smoothing
  Alpha.focus   = 0.06;       % Bandwidth Focus Smoothing
  MinPower      = 0.12;
  NoisePerScan  = 0.18;     

% Bandwidth Focusing Parameters
  BWTotal       = 1.0;
  FocusHalfWinR = 9;      
  FocusHalfWinD = 9;      
  Kcenters      = 4;  

% Coupling Strength 
  GainBWToSNR     = 0.25;  % SNR multiplier: 1 + gain*(eff_local - 0.5)
  GainBWToClutter = 0.25;  % Clutter variance reduction: (1 - gain*(effBW - 0.5))
  GainBWToHrange  = 0.40;  % Range narrowing: hr0 / (1 + gain*(eff_local - 0.5))

PowerMapRD = ones(NumRangeBins, NumDoplerBins);        % When attention >= min_power
BWFocusRD  = zeros(NumRangeBins, NumDoplerBins);       % Attention focus
DwellRD    = ones(NumRangeBins, NumDoplerBins);        % dwell (integration-time bias)

DetLog = false(NumScans, NumRangeBins, NumDoplerBins);
SigLog = zeros(NumScans, NumRangeBins, NumDoplerBins);

TP=0; FP=0; TN=0; FN=0;

% Current States
  CarRange   = Car.InitialRange; 
  CarDoppler = Car.DopplerBin;
  PedRange   = Pedestrian.InitialRange;   
  PedDoppler = Pedestrian.DopplerBin;

figure('Name','Cognitive Processing','Color','w','Position',[80 60 1200 900]);
gifFilename = 'CognitiveRadar.gif';  
gifDelay    = 0.08;          
FrameCount  = 0;    

for scan = 1:NumScans

    EffectiveBW = BWTotal*(0.5 + 0.5*BWFocusRD);

    Noise   = max(0, NoiseMu + 0.15*randn(NumRangeBins, NumDoplerBins));

    Clutter = ClutterScale*abs((randn(NumRangeBins,NumDoplerBins)+1i*randn(NumRangeBins,NumDoplerBins))/sqrt(2));
    Clutter = Clutter .* (1.0 - GainBWToClutter*(EffectiveBW - 0.5));
    RD      = Noise + Clutter;

    % Interference
      if rand < InterfProb
         if rand < 0.5
            rC          = randi([8, NumRangeBins-8]); width = randi(InterfWidth);
            rIdx        = max(1,rC-width):min(NumRangeBins,rC+width);
            RD(rIdx, :) = RD(rIdx, :) + InterfPower*(0.5+0.5*rand);
        else
            dC          = randi([8, NumDoplerBins-8]); width = randi(InterfWidth);
            dIdx        = max(1,dC-width):min(NumDoplerBins,dC+width);
            RD(:, dIdx) = RD(:, dIdx) + InterfPower*(0.5+0.5*rand);
        end
      end

    % Update target states
      CarRange = CarRange + Car.Velocity;   CarDoppler = CarDoppler + Car.VelocityDrift;
      PedRange = PedRange + Pedestrian.Velocity; 
      CarRange = max(4, min(NumRangeBins-3, CarRange));
      PedRange = max(4, min(NumRangeBins-3, PedRange));

    % BW sampling around each target center
      win = 2;
      [CarRngIdx, CarDopIdx] = deal(round(CarRange), round(CarDoppler));
      [PedRngIdx, PedDopIdx] = deal(round(PedRange), round(PedDoppler));

      CarEffLocal = localMean(EffectiveBW, CarRngIdx, CarDopIdx, win);
      PedEffLocal = localMean(EffectiveBW, PedRngIdx, PedDopIdx, win);

      CarSNRBoost = 1 + GainBWToSNR*(CarEffLocal - 0.5);
      PedSNRBoost = 1 + GainBWToSNR*(PedEffLocal - 0.5);

      CarHREff = Car.hr0 / (1 + GainBWToHrange*(CarEffLocal - 0.5));
      PedHREff = Pedestrian.hr0 / (1 + GainBWToHrange*(PedEffLocal - 0.5));
      CarHREff = max(0.8, CarHREff); 
      PedHREff = max(0.7, PedHREff);

    % Car
      SNRlinCar = 10^((Car.SNRdB)*CarSNRBoost/10);
      RD        = RD + SNRlinCar * 0.9 * gaussPatch2D(NumRangeBins, NumDoplerBins, CarRngIdx, CarDopIdx, CarHREff, Car.hd0).* PowerMapRD .* (0.6 + 0.4*DwellRD);

    % Pedestrian
      DopCenter = PedDoppler + round(3.5*sin(2*pi*Pedestrian.GaitFreq*scan));
      SideAmp   = 0.65 + 0.35*sin(2*pi*Pedestrian.GaitFreq*scan + pi/5); % time-varying spread
      SNRlinPed = 10^((Pedestrian.SNRdB)*PedSNRBoost/10);
      RD        = RD + SNRlinPed * 0.7 * gaussPatch2D(NumRangeBins, NumDoplerBins, PedRngIdx, DopCenter, PedHREff, Pedestrian.hd0) .* PowerMapRD .* (0.6 + 0.4*DwellRD);
      for sb = 1:Pedestrian.SidebandW
          w  = SideAmp * exp(-0.5*(sb/ (Pedestrian.SidebandW/1.8))^2);
          RD = RD + SNRlinPed * 0.18*w * gaussPatch2D(NumRangeBins, NumDoplerBins, PedRngIdx, DopCenter+sb, max(0.9,PedHREff), Pedestrian.hd0) .* PowerMapRD;
          RD = RD + SNRlinPed * 0.18*w * gaussPatch2D(NumRangeBins, NumDoplerBins, PedRngIdx, DopCenter-sb, max(0.9,PedHREff), Pedestrian.hd0) .* PowerMapRD;
      end

      RD = RD .* (0.9 + 0.2*EffectiveBW); 

      SigLog(scan,:,:) = RD;

    % CA-CFAR
      Detections       = CA_CFAR_AP(RD, CFAR, [NumRangeBins, NumDoplerBins]);
      DetLog(scan,:,:) = Detections;

    % Saliency from CFAR (current + recent), Smoothed
      RecentWin = max(1, scan-5):scan;
      DetRecent = squeeze(sum(DetLog(RecentWin,:,:),1)) > 0;
      Saliency  = imgaussfilt(double(Detections) + 0.5*double(DetRecent), 2.0);

    % Select top-K focus centres from saliency
      [~, idxSort] = sort(Saliency(:), 'descend');
      idxSort      = idxSort(1:min(Kcenters, numel(idxSort)));
      [rC, dC]     = ind2sub([NumRangeBins,NumDoplerBins], idxSort);
      centers = [rC(:), dC(:)];
      if max(Saliency(:)) < 0.25 && scan < 5
         centers = [CarRngIdx, CarDopIdx; PedRngIdx, DopCenter];
      end

    % Update BW focus
      LocalF    = fusedFocus2D(NumRangeBins, NumDoplerBins, centers, FocusHalfWinR, FocusHalfWinD);
      BWFocusRD = (1 - Alpha.focus)*BWFocusRD + Alpha.focus*LocalF;

    % Attention PowerMap, reward from detections only
      Reward     = mat2gray(double(Detections)); % [0..1]
      PowerMapRD = (1 - Alpha.power)*PowerMapRD + Alpha.power*(1 + Reward);
      PowerMapRD = max(PowerMapRD, MinPower);
      PowerMapRD = PowerMapRD / max(PowerMapRD(:));
      PowerMapRD = MinPower + (1 - MinPower)*PowerMapRD;

    dwellBase = mat2gray(PowerMapRD + 0.6*Saliency);
    DwellRD   = 0.25 + 0.75*dwellBase;
    if rand < NoisePerScan
        DwellRD = max(DwellRD, rand(NumRangeBins,NumDoplerBins));
    end

    GT                       = false(NumRangeBins,NumDoplerBins);
    GT(CarRngIdx, CarDopIdx) = true;
    GT(PedRngIdx, DopCenter) = true;
    GTd                      = imdilate(GT, ones(3,3));
   
    TP = TP + sum(Detections & GTd, 'all');
    FP = FP + sum(Detections & ~GTd, 'all');
    TN = TN + sum(~Detections & ~GTd, 'all');
    FN = FN + sum(~Detections &  GTd, 'all');

    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

    nexttile(1);
    imagesc(RD); axis xy; title(sprintf('Scan %d: Range Doppler', scan));set(gca,'fontsize',14)
    xlabel('Doppler Bin'); ylabel('Range Bin'); colorbar;colormap('jet');%clim([0 11])

    nexttile(2);
    imagesc(Detections); axis xy; title('CFAR Detections');set(gca,'fontsize',14)
    xlabel('Doppler Bin'); ylabel('Range Bin'); colorbar; colormap('gray');%clim([0 1])

    nexttile(3);
    imagesc(EffectiveBW); axis xy; title('Cognitive Focus');set(gca,'fontsize',14)
    xlabel('Doppler Bin'); ylabel('Range Bin'); colorbar;colormap('jet');%clim([0.5 0.8])

    nexttile(4);
    imagesc(PowerMapRD); axis xy; title('Attention Power Map');set(gca,'fontsize',14)
    xlabel('Doppler Bin'); ylabel('Range Bin'); colorbar;colormap('jet'); clim([0.8 1])

    drawnow;

    frame = getframe(gcf);
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);
    if scan == 1
        imwrite(imind, cm, gifFilename, 'gif', 'Loopcount', inf, 'DelayTime', gifDelay);
    else
        imwrite(imind, cm, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', gifDelay);
    end
    FrameCount = FrameCount + 1;

end

P_D  = TP / (TP + FN + eps);
P_FA = FP / (FP + TN + eps);
fprintf('P_D: %.3f\n', P_D);
fprintf('P_FA: %.3f\n', P_FA);

%% FUNCTIONS
function det = CA_CFAR_AP(X, CFAR, sz)
    Nr = sz(1); Nd = sz(2);
    det = false(Nr,Nd);

    Tr = CFAR.TrainR; 
    Td = CFAR.TrainD;
    Gr = CFAR.GuardR;
    Gd = CFAR.GuardD;
    k  = CFAR.k;

    Sp              = zeros(Nr+1, Nd+1);
    Sp(2:end,2:end) = cumsum(cumsum(X,1),2);

    function s = rectsum(a,b,c,d)
        a = max(1, min(Nr, a));
        b = max(1, min(Nr, b));
        c = max(1, min(Nd, c)); 
        d = max(1, min(Nd, d));
        if a>b || c>d
           s = 0; 
           return;
        end
        s = Sp(b+1,d+1) - Sp(a,d+1) - Sp(b+1,c) + Sp(a,c);
    end

    for r = 1:Nr
        for d = 1:Nd
            % Full (training+guard+CUT) window
            r1 = r - (Tr+Gr); 
            r2 = r + (Tr+Gr);
            d1 = d - (Td+Gd);
            d2 = d + (Td+Gd);

            % Guard+CUT
            g1 = r - Gr; 
            g2 = r + Gr;
            h1 = d - Gd;
            h2 = d + Gd;

            totSum = rectsum(r1, r2, d1, d2);
            rr1    = max(1,r1); rr2 = min(Nr,r2);
            dd1    = max(1,d1); dd2 = min(Nd,d2);
            totNum = (rr2-rr1+1) * (dd2-dd1+1);

            guardSum = rectsum(g1, g2, h1, h2);
            gg1      = max(1,g1); gg2 = min(Nr,g2);
            hh1      = max(1,h1); hh2 = min(Nd,h2);
            guardNum = (gg2-gg1+1) * (hh2-hh1+1);

            refNum = totNum - guardNum;
            if refNum <= 0
                det(r,d) = false; 
                continue;
            end

            refMean  = (totSum - guardSum) / refNum;
            thr      = refMean * (1 + 0.25*k);
            det(r,d) = X(r,d) > thr;
        end
    end
end

% Normalized 2D Gaussian patch centered at (r0,d0)
function W = gaussPatch2D(Nr, Nd, r0, d0, hr, hd)
    [rr, dd] = ndgrid(1:Nr, 1:Nd);
    W        = exp(-((rr-r0).^2)/(2*hr^2) - ((dd-d0).^2)/(2*hd^2));
    W        = W / max(W(:) + eps);
end

% Focus map from multiple (r,d) centres
function F = fusedFocus2D(Nr, Nd, centers, hr, hd)
    F = zeros(Nr, Nd);
    for k = 1:size(centers,1)
        F = max(F, gaussPatch2D(Nr, Nd, centers(k,1), centers(k,2), hr, hd));
    end
    if max(F(:)) > 0
       F = F / max(F(:));
    end
end

% Local mean in a square (2*win+1)^2 window
function m = localMean(A, r0, d0, win)
    [Nr, Nd] = size(A);
    r1       = max(1, r0-win); r2 = min(Nr, r0+win);
    d1       = max(1, d0-win); d2 = min(Nd, d0+win);
    patch    = A(r1:r2, d1:d2);
    m        = mean(patch(:));
end


