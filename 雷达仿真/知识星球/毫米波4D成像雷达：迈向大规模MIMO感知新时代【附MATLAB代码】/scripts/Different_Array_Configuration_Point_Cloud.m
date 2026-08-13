%%  PreProcessing
clear, close all
clc
%%  Initialization

addpath('../data/')
addpath(genpath('../functions/'))

%%  4D Imaging Radar Parameters

c                       = physconst('LightSpeed');
fc                      = 77e9;
lambda                  = c/fc;
Tc                      = 50e-6;
rangeMax                = 150;
rangeRes                = 0.25;
bw                      = c/(2*rangeRes);
sweepSlope              = bw/Tc;
fbeatMax                = 2*rangeMax*bw/c;
fifMax                  = sweepSlope*2*rangeMax/c;
fs                      = 2 * fifMax;
samples                 = 2 * fbeatMax;

%%  Virtual Array (lambda units)

load vaPos_12x16
load vaPos_48x48
load vaPos_16x16_1
load vaPos_16x16_2

useArray = '12x16';
switch useArray
    case '12x16',   VA = vaPos_12x16;
    case '48x48',   VA = vaPos_48x48;
    case '16x16_1', VA = vaPos_16x16_1;
    case '16x16_2', VA = vaPos_16x16_1;
end
vaX = VA(:,1);
vaY = VA(:,2);

xL      = unique(round(vaX,6));
yL      = unique(round(vaY,6));
pitchX  = min(diff(xL));
pitchY  = min(diff(yL));
Ngx     = round((max(xL)-min(xL))/pitchX) + 1;
Ngy     = round((max(yL)-min(yL))/pitchY) + 1;
gx      = round((vaX - min(vaX))/pitchX) + 1;
gy      = round((vaY - min(vaY))/pitchY) + 1;
gLin    = sub2ind([Ngx Ngy], gx, gy);

%%  Scaling

ScaleFactor             = 30;
resScale                = 4;

%%  Antenna Positioning

antennaCenter           = [0 2500 1000] / ScaleFactor;
antennaDir              = [0 -1 0];
antennaXDir             = [1 0 0];

%%  Target (Man Walking in Circle)

load dataFrame

dataFrameX              = dataFrame(:,:,1);
dataFrameY              = dataFrame(:,:,3);
dataFrameZ              = dataFrame(:,:,2);

frameNo                 = 2e3;
dataFrame               = cat(3,dataFrameX,dataFrameY,dataFrameZ) / ScaleFactor;
SlectedFrame            = squeeze(dataFrame(frameNo,:,:));
tarCenter               = mean(SlectedFrame);

lclCoordSlcFrame        = resScale * global2localcoord(SlectedFrame.', 'rr', tarCenter.');
tarPos                  = local2globalcoord(lclCoordSlcFrame, 'rr', tarCenter.').';

%%  Geometry

distVec             = tarPos - antennaCenter;
R                   = pdist2(tarPos, antennaCenter);
theta               = 90 - acosd(distVec * antennaDir.' ./ R);
projectionVecs      = distVec - antennaDir .* (distVec * antennaDir.');
idx                 = find(antennaDir == 0);
phi                 = atan2d(projectionVecs(:,idx(end)), projectionVecs(:,idx(1))) ...
    - atan2d(antennaXDir(idx(end)), antennaXDir(idx(1)));

%%  Processing Signal  (per-element synthesis using the real VA)

tP                  = (0:(samples-1)).'./fs;                                % samples x 1
td                  = 2 * R / c;                                            % Nt x 1

% Direction cosines
uTar                = cosd(phi).*cosd(theta);                                % Nt x 1
vTar                = sind(phi).*cosd(theta);                                % Nt x 1

% Per-element spatial phase (positions already in lambda -> no /lambda)
spatialPhase        = 2*pi*( vaX*uTar.' + vaY*vTar.' );                      % Nva x Nt
beatPhase           = 2*pi*sweepSlope * (tP * td.');                         % samples x Nt
prSigList           = exp(1i*beatPhase) * exp(1i*spatialPhase).';            % samples x Nva

% Place elements onto the lattice (holes = zeros) for the FFT
prSig               = zeros(samples, Ngx, Ngy);
for s = 1:samples
    tmp             = zeros(Ngx, Ngy);
    tmp(gLin)       = prSigList(s, :);
    prSig(s, :, :)  = tmp;
end

%%  Image Reconstruction

rangeBinAxis        = ((1:samples/2) - 1) / samples .* fs * c / 2 / sweepSlope;

fnT = (-1/2:1/Ngx:1/2-1/Ngx).' / pitchX;          % x dir-cosine axis
fnR = (-1/2:1/Ngy:1/2-1/Ngy)  / pitchY;           % y dir-cosine axis

NtInt = 500;
NrInt = 500;

fnTInt = (-1/2:1/NtInt:1/2-1/NtInt).' / pitchX;
fnRInt = (-1/2:1/NrInt:1/2-1/NrInt)  / pitchY;

[FNRINT, FNTINT] = meshgrid(fnRInt, fnTInt);
[FNR, FNT] = meshgrid(fnR, fnT);

prSigF              = fftn(prSig);

[~, locsR]          = findpeaks(mean(abs(prSigF(1:samples/2,:,:)), [2 3]));
rEst                = rangeBinAxis(locsR);

AngleFFTshift       = squeeze(fftshift(fftshift(abs(prSigF), 2),3));

lclCoordCell        = cell(1,length(rEst));
for i = 1:length(rEst)

    Fq = interp2(FNR, FNT, squeeze(AngleFFTshift(locsR(i),:,:)), FNRINT, FNTINT, 'makima');
    [~, locX, locY] = peaks2(Fq, 'MinPeakHeight', 0.9 * max(Fq, [], 'all'));

    if ~isempty(locX)

        fnTd = (locX(:)/NtInt - 1/2) / pitchX;
        fnRd = (locY(:)/NrInt - 1/2) / pitchY;

        thetaEst = acosd(sqrt(fnTd.^2 + fnRd.^2));
        phiEst   = atan2d(fnRd, fnTd);

        AngleRealFlag = imag(phiEst) <= 1e-6 & imag(thetaEst) <= 1e-6;

        lclCoordCell{i} = [phiEst(AngleRealFlag).'; thetaEst(AngleRealFlag).'; ...
                           repmat(rEst(i), 1, sum(AngleRealFlag))];

    end

end

lclCoord            = cell2mat(lclCoordCell(~cellfun(@isempty, lclCoordCell)));
Coord               = local2globalcoord(lclCoord,'sr',antennaCenter.',[1 0 0; 0 0 -1; 0 1 0]);

CoordlclNormal      = global2localcoord(Coord, 'rr', tarCenter.') / resScale;
CoordNormal         = ScaleFactor * local2globalcoord(CoordlclNormal, 'rr', tarCenter.');

% - Recenter x-y to (0,0); leave z unchanged
xyCenter            = mean(CoordNormal(1:2,:), 2);
CoordNormal(1:2,:)  = CoordNormal(1:2,:) - xyCenter;

%%  Ground Truth (normalized in the same way as CoordNormal)

gtLclNormal         = global2localcoord(tarPos.', 'rr', tarCenter.') / resScale;
gtNormal            = ScaleFactor * local2globalcoord(gtLclNormal, 'rr', tarCenter.');

gtXYCenter          = mean(gtNormal(1:2,:), 2);
gtNormal(1:2,:)     = gtNormal(1:2,:) - gtXYCenter;

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
    gHndl.XTick         = -750:250:750;
    gHndl.YTick         = -750:250:750;
    gHndl.ZTick         = 0:500:2000;
    gHndl.XTickLabel    = mat2cell(gHndl.XTick / 1e3,1,length(gHndl.XTick));
    gHndl.YTickLabel    = mat2cell(gHndl.YTick / 1e3,1,length(gHndl.YTick));
    gHndl.ZTickLabel    = mat2cell(gHndl.ZTick / 1e3,1,length(gHndl.ZTick));
    axis(gHndl,'equal')
    axis(gHndl,[-750 750 -750 750 0 2000])
    grid(gHndl,'on')
    box(gHndl,'on')
    hold(gHndl,'on')
    view(gHndl,viewMat(i,:))
    xlabel('$x$ (m)','FontSize',11,'Interpreter','Latex')
    ylabel('$y$ (m)','FontSize',11,'Interpreter','Latex')
    zlabel('$z$ (m)','FontSize',11,'Interpreter','Latex')

    % Reconstruction
    scatter3(CoordNormal(1,:), CoordNormal(2,:), CoordNormal(3,:), 18, 'filled');

    % Transparent ground truth
    scatter3(gtNormal(1,:), gtNormal(2,:), gtNormal(3,:), ...
             18, [0.8500 0.3250 0.0980], 'filled', ...
             'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);

end