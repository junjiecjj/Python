%%  PreProcessing
clear, close all
clc
%%  Initialization

addpath('../data/')
addpath(genpath('../functions/'))

%%  4D Imaging Radar Parameters

c                       = physconst('LightSpeed');                                  % Speed of Light
fc                      = 77e9;                                                     % Radar Center Frequency
lambda                  = c/fc;                                                     % Radar Wavelength
Tc                      = 50e-6;                                                    % FMCW Chirp Time
rangeMax                = 150;                                                      % Mmaximum Detection Range
vMax                    = 250*1000/3600;                                            % Maximum Velocity of Cars (m/s)
rangeRes                = 0.25;                                                     % Range Resolution (It will Reduce to 12 cm)
bw                      = c/(2*rangeRes);                                           % Sweep Bandwidth
sweepSlope              = bw/Tc;                                                    % Sweep Slope
fdopMax                 = 2*vMax/lambda;                                            % Maximum Doppler shift (Hz)
fbeatMax                = 2*rangeMax*bw/c;                                          % Maximum Beat Frequency (Hz)
fifMax                  = sweepSlope*2*rangeMax/c;                                  % Maximum Received IF (Hz)
fs                      = 2 * fifMax;                                               % Sampling Rate
samples                 = 2 * fbeatMax;                                             % Samples in One Chirp
Nt                      = 128;                                                      % No. of Transmitting Antenna
Nr                      = 128;                                                      % No. of Receiving Antenna
dx                      = lambda/2;                                                 % Transmitting Antenna Sapcing
dy                      = lambda/2;                                                 % Receiving Antenna Sapcing

%%  Scaling

ScaleFactor             = 30;                                                       % Data Scaling Factor
resScale                = [4 2 2];                                                  % Increasing Renge Resolution Artificially

%%  Antenna Positioning

% - Placing Antenna in x-z Plane
antennaCenter           = [0 2500 1000] / ScaleFactor;
antennaDir              = [0 -1 0];
antennaXDir             = [1 0 0];

%%  Targets Properties

% - Man Walking in Circle
load dataFrame

dataFrameX              = dataFrame(:,:,1);
dataFrameY              = dataFrame(:,:,3);
dataFrameZ              = dataFrame(:,:,2);

frameNo                 = 2e3;                                                      % No. of Frame Snapshot
dataFrame               = cat(3,dataFrameX,dataFrameY,dataFrameZ) / ScaleFactor;    % Scaling Data
SlectedFrame            = squeeze(dataFrame(frameNo,:,:));                          % Frame Snapshot
dataFrameCenter         = mean(SlectedFrame);                                       % Frame Center

% - Local Coordinate of the Selected Frame
lclCoordSlcFrame        = resScale(1) * global2localcoord(SlectedFrame.', 'rr', dataFrameCenter.');

% - Reconstructing the Selected Frame Data
dataFrame               = local2globalcoord(lclCoordSlcFrame, 'rr', dataFrameCenter.');
dataFrame               = dataFrame.';

% - Pyramid
BaseCenterPyr           = [2750, 1250, 0];
ScalePyr                = resScale(2) * 500;
[xPyr, yPyr, zPyr]      = PyramidSurf(4, ScalePyr, BaseCenterPyr);
PyrData                 = [xPyr yPyr zPyr] / ScaleFactor;

% - Sphere
rSph                    = resScale(3) * 500;
CenterSph               = [-2500 1000 rSph/resScale(3)];

[Theta, Phi]            = meshgrid(linspace(-pi/2,pi/2,5), linspace(0,2*pi,10));

xSph                    = rSph * cos(Theta(:)) .* cos(Phi(:)) + CenterSph(1);
ySph                    = rSph * cos(Theta(:)) .* sin(Phi(:)) + CenterSph(2);
zSph                    = rSph * sin(Theta(:)) + CenterSph(3);

SphData                 = [xSph ySph zSph] / ScaleFactor;

%%  Targets Parameter Defining

% - Pre Allocating
CoordNormal             = cell(3,1);
objCenter               = cell(3,1);
objPos                  = cell(3,1);

% - Man Walking in Cycle
objPos{1}               = dataFrame;
objCenter{1}            = dataFrameCenter;

% - Pyramid
objPos{2}               = PyrData;
objCenter{2}            = BaseCenterPyr / ScaleFactor;

% - Sphere
objPos{3}               = SphData;
objCenter{3}            = CenterSph / ScaleFactor;

%%  Main Loop

for objCnt = 1:length(objPos)

    objData = objPos{objCnt};

    distVec             = objData - antennaCenter;
    R                   = pdist2(objData,antennaCenter);
    theta               = 90 - acosd(distVec * antennaDir.' ./ R);
    projectionVecs      = distVec - antennaDir .* (distVec * antennaDir.');
    idx                 = find(antennaDir == 0);
    phi                 = atan2d(projectionVecs(:,idx(end)), projectionVecs(:,idx(1))) ...
        - atan2d(antennaXDir(idx(end)), antennaXDir(idx(1)));

    R0                  = permute(R.',[1,3,4,2]);
    phi0                = permute(phi.',[1,3,4,2]);
    theta0              = permute(theta.',[1,3,4,2]);

    %%  Processing Signal

    tP                  = (0:(samples-1)).'./fs;
    nT                  = permute((1:Nt),[1,2,3]);
    nR                  = permute((1:Nr),[1,3,2]);

    td                  = 2 * R0/ c;
    ax                  = exp(2*pi*1i*dx/lambda*cosd(phi0).*cosd(theta0) .* (nT - 1));
    ay                  = exp(2*pi*1i*dy/lambda*sind(phi0).*cosd(theta0) .* (nR - 1));

    prSig               = sum(ax .* ay .* exp(2*pi*1i* sweepSlope*td .* tP), 4);

    %%  Image Reconstruction

    rangeBinAxis        = ((1:samples/2) - 1) / samples .* fs * c / 2 / sweepSlope;

    if Nt < 64 || Nr < 64

        fnT = (-1/2:1/Nt:1/2-1/Nt).' * lambda / dx;
        fnR = (-1/2:1/Nr:1/2-1/Nr) * lambda / dy;

        NtInt = 500;
        NrInt = 500;

        fnTInt = (-1/2:1/NtInt:1/2-1/NtInt).' * lambda / dx;
        fnRInt = (-1/2:1/NrInt:1/2-1/NrInt) * lambda / dy;

        [FNRINT, FNTINT] = meshgrid(fnRInt, fnTInt);
        [FNR, FNT] = meshgrid(fnR, fnT);

    end

    prSigF              = fftn(prSig);

    [~, locsR]          = findpeaks(mean(abs(prSigF(1:samples/2,:,:)), [2 3]));
    rEst                = rangeBinAxis(locsR);

    AngleFFTshift       = squeeze(fftshift(fftshift(abs(prSigF), 2),3));

    lclCoordCell        = cell(1,length(rEst));
    for i = 1:length(rEst)

        if Nt < 64 || Nr < 64

            Fq = interp2(FNR, FNT, squeeze(AngleFFTshift(locsR(i),:,:)), FNRINT, FNTINT, 'makima');
            [~, locX, locY] = peaks2(Fq, 'MinPeakHeight', 0.9 * max(Fq, [], 'all'));

        else

            [~, locX, locY] = peaks2(squeeze(AngleFFTshift(locsR(i),:,:)), 'MinPeakHeight', 0.9 * max(AngleFFTshift(locsR(i),:,:), [], 'all'));

        end


        if ~isempty(locX)

            if Nt >= 64 && Nr >= 64

                NtInt = Nr;
                NrInt = Nr;

            end

            fnT = (locX(:)/NtInt - 1/2) * lambda / dx;
            fnR = (locY(:)/NrInt - 1/2) * lambda / dy;

            thetaEst = acosd(sqrt(fnT.^2 + fnR.^2));
            phiEst = atan2d(fnR,fnT);

            AngleRealFlag = imag(phiEst) <= 1e-6 & imag(thetaEst) <= 1e-6;

            lclCoordCell{i} = [phiEst(AngleRealFlag).'; thetaEst(AngleRealFlag).'; repmat(rEst(i), 1, sum(AngleRealFlag))];
            
        end

    end

    emptyFlag           = cellfun(@(x) isempty(x), lclCoordCell);

    lclCoord            = cell2mat(lclCoordCell(~emptyFlag));
    Coord               = local2globalcoord(lclCoord,'sr',antennaCenter.',[1 0 0; 0 0 -1; 0 1 0]);

    % - Rescaling to Normal Coordinates
    CoordlclNormal      = global2localcoord(Coord, 'rr', objCenter{objCnt}.') / resScale(objCnt);
    CoordNormal{objCnt} = ScaleFactor * local2globalcoord(CoordlclNormal, 'rr', objCenter{objCnt}.');

end

%%  Plot Results

ScreenSize              = get(0,'ScreenSize');

viewMat                 = [-37.5 30; 0 0; 0 90];
fhdl                    = gobjects(size(viewMat,1), 1);
figureName              = {'(3D View)', '(X-Z View)', '(X-Y View)'};
figureLoc               = {'center', 'west', 'east'};

for i = 1:size(viewMat,1)

    fhdl(i)             = figure('Name',['4D Image Radar Output - ', figureName{i}],'NumberTitle','off',...
        'Position', [0 0 floor(ScreenSize(3)/3) floor(ScreenSize(4)/3)]);
    movegui(figureLoc{i})

    gHndl               = gca;
    gHndl.NextPlot      = 'replacechildren';
    gHndl.XTick         = -3500:1000:3500;
    gHndl.YTick         = -3500:1000:3500;
    gHndl.ZTick         = 0:1000:2000;
    gHndl.XTickLabel    = mat2cell(gHndl.XTick / 1e3,1,length(gHndl.XTick));
    gHndl.YTickLabel    = mat2cell(gHndl.YTick / 1e3,1,length(gHndl.YTick));
    gHndl.ZTickLabel    = mat2cell(gHndl.ZTick / 1e3,1,length(gHndl.ZTick));
    axis(gHndl,'equal')
    axis(gHndl,[-3500 3500 -3500 3500 0 2000])
    grid(gHndl,'on')
    box(gHndl,'on')
    hold(gHndl,'on')
    view(gHndl,viewMat(i,:))
    xlabel('$x$ (m)','FontSize',11,'Interpreter','Latex')
    ylabel('$y$ (m)','FontSize',11,'Interpreter','Latex')
    zlabel('$z$ (m)','FontSize',11,'Interpreter','Latex')

    for objCnt = 1:length(objPos)
        plot3(CoordNormal{objCnt}(1,:),CoordNormal{objCnt}(2,:),CoordNormal{objCnt}(3,:), '.','MarkerSize',12)
    end

end



