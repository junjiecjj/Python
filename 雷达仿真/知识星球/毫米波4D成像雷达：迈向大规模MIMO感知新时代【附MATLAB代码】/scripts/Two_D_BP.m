clear
close all
clc

%%  Common Parameters

c       = 3e8;
fc      = 77e9;
lambda  = c / fc;
dt      = lambda / 2;

thetaVec    = (0:90)';      % elevation
phiVec      = 0:360;        % azimuth

R           = cat(3, sind(thetaVec).*cosd(phiVec), ...
                     sind(thetaVec).*sind(phiVec), ...
                     cosd(thetaVec) + 0*phiVec);

% Steering direction
thetaS      = 30;
phiS        = 60;
Rw          = cat(3, sind(thetaS)*cosd(phiS), ...
                     sind(thetaS)*sind(phiS), ...
                     cosd(thetaS));

%%  Loop over Configurations

mList       = [12 32];
figureLoc   = {'west', 'east'};
fhdl        = gobjects(length(mList), 1);
ScreenSize  = get(0,'ScreenSize');

for k = 1:length(mList)

    %%  URA Definition

    mX = mList(k);
    mY = mList(k);
    Nt = mX * mY;

    txX         = ((1:mX) - (mX+1)/2) * dt;
    txY         = ((1:mY) - (mY+1)/2) * dt;
    [pX, pY]    = meshgrid(txX, txY);
    pZ          = zeros(size(pX));
    P           = cat(3, pX, pY, pZ);

    %%  Beampattern Computation

    tau         = -1/c .* sum(permute(P,[4,5,3,1,2]) .* R, 3);
    APrime      = reshape(exp(1i*2*pi/lambda*c.*tau), length(thetaVec), length(phiVec), []);
    A           = conj(permute(reshape(APrime, [], Nt), [3, 2, 1]));

    tauW        = -1/c .* sum(permute(P,[4,5,3,1,2]) .* Rw, 3);
    w           = squeeze(reshape(exp(1i*2*pi/lambda*c.*tauW), 1, 1, []));

    Pattern     = reshape(squeeze(abs(pagemtimes(A,w)).^2), ...
                          [length(thetaVec) length(phiVec)]);

    %%  2D Beampattern Plot

    patt            = Pattern;
    min_threshold   = max(patt(:)) * 1e-3;
    patt(patt < min_threshold) = min_threshold;

    fhdl(k) = figure('Name', sprintf('2D Beampattern %dx%d URA', mX, mY), ...
        'NumberTitle','off', ...
        'Position', [0 0 floor(ScreenSize(3)/3) floor(ScreenSize(4)/3)]);
    movegui(figureLoc{k})

    imagesc(thetaVec, phiVec, 10*log10(patt.'))
    set(gca, 'YDir', 'normal')
    xlabel('Elevation (degree $^\circ$)', 'Interpreter', 'latex', 'FontSize', 12)
    ylabel('Azimuth (degree $^\circ$)', 'Interpreter', 'latex', 'FontSize', 12)
    colorbar
    xlim([0 90])
    ylim([0 360])
    xticks(0:30:90)
    yticks(0:90:360)
    box on

end