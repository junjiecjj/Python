% MIMO FMCW Radar Simulation - "4D" point cloud
% Sweep: B = [150 MHz, 1 GHz, 4 GHz] x Lva = [16 64 192 576]

%%  PreProcessing
clear, close all
clc

addpath('../data/')

%%  Importing Image ("4D" only)

originalImage   = flip(imread('4D.png'),1);
binaryImage     = imbinarize(im2gray(originalImage));

dsRate          = 10;
lpp             = 0.006;

dsImage         = downsample(binaryImage,dsRate);
dsImage         = downsample(dsImage.',dsRate);

usImage         = upsample(~dsImage,dsRate);
usImage         = ~upsample(usImage.',dsRate);

[row, col]      = ind2sub(size(usImage), find(usImage(:) == 0));

x               = col * lpp;
x               = abs((max(x) - min(x)) / 2) + x;
y               = row * lpp;
y               = y - abs((max(y) - min(y)) / 2);

[phi, r]        = cart2pol(x,y);

%%  Radar Parameters

c               = physconst('LightSpeed');
fc              = 77e9;
Tc              = 50e-6;
rangeMax        = 10;
lambda          = c/fc;
d               = lambda/2;
NTx             = 1;

bwList          = [150e6 1e9 4e9];
NRxList         = [16 64 192 576];

%%  Simulation Sweep

nBw             = length(bwList);
nRx             = length(NRxList);

map             = cell(nBw, nRx);
xGridCell       = cell(nBw, nRx);
yGridCell       = cell(nBw, nRx);

for iBw = 1:nBw
    for iRx = 1:nRx

        bw              = bwList(iBw);
        NRx             = NRxList(iRx);
        NVr             = NTx * NRx;

        slope           = bw/Tc;
        fifMax          = slope*2*rangeMax/c;
        fs              = 2 * fifMax;
        samples         = Tc * fs;

        % - Target Parameters
        range           = permute(r.',[1,3,4,5,2]);
        angle           = permute(rad2deg(phi).',[1,3,4,5,2]);

        tP              = (0:(samples-1)).'./fs;
        i               = permute((1:NVr),[1,3,2]);
        td              = 2 * range / c;

        a               = 2*pi*slope*td.*tP + 2*pi*d*i.*sind(angle)/lambda;

        mixerTargets    = exp(1j*a);
        mixer           = sum(mixerTargets,5);

        radarCube       = squeeze(sum(mixer,4));

        % - FFT Processing
        nFFTrange       = 2^nextpow2(size(radarCube,1)+1);
        nFFTangle       = 2^nextpow2(size(radarCube,2)+1);

        FFT             = abs(fftshift(fft2(radarCube, nFFTrange, nFFTangle),2));
        validFFT        = FFT(1:nFFTrange/2,:);

        rangeAxis       = linspace(0,rangeMax,nFFTrange/2);
        angleAxis       = asind(2*(-1/2:1/nFFTangle:1/2));

        map{iBw,iRx}    = circshift(padarray(validFFT,[0 1],'replicate','pre'), -1, 2);

        [R, Phi]        = meshgrid(rangeAxis,angleAxis);
        [xGrid, yGrid]  = pol2cart(deg2rad(Phi),R);
        xGridCell{iBw,iRx} = xGrid;
        yGridCell{iBw,iRx} = yGrid;

        fprintf('Done: B = %g MHz, Lva = %d\n', bw/1e6, NVr);

    end
end

%%  Plot Results (all 12, table layout: rows = B, columns = Lva)

ScreenSize      = get(0,'ScreenSize');
titleBarH       = 85;
panelH          = floor(ScreenSize(4)/nBw) - titleBarH;
panelW          = floor(panelH/2);

for iBw = 1:nBw
    for iRx = 1:nRx

        figure('Name', sprintf('B = %g MHz, Lva = %d', ...
            bwList(iBw)/1e6, NTx*NRxList(iRx)), 'NumberTitle','off', ...
            'Position', [50 + (iRx-1)*(panelW+15), ...
                         ScreenSize(4) - iBw*(panelH+titleBarH), ...
                         panelW, panelH]);

        surf(xGridCell{iBw,iRx}, yGridCell{iBw,iRx}, ...
             map{iBw,iRx}.', 'EdgeColor','none')
        view(0,90)
        axis equal
        xlim([0, rangeMax])
        ylim([-rangeMax, rangeMax])
        xticks(0:2:rangeMax)
        yticks(-rangeMax:5:rangeMax)
        box on
        grid on

    end
end