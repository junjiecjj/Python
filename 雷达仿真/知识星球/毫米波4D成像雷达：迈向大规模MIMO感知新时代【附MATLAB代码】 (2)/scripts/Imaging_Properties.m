%%  PreProcessing
clear, close all
clc
%%  Initialization

addpath('../data/')
addpath(genpath('../functions/'))

ScreenSize                  = get(0,'ScreenSize');
handle                      = [];

% - Man Walking in Circle
load dataFrame
frameNo                     = 2e3;

% - Pyramid
BaseCenterPyr               = [2750, 1250, 0];
ScalePyr                    = 500;
[xPyr, yPyr, zPyr]          = PyramidSurf(4, ScalePyr, BaseCenterPyr);

% - Sphere
rSph                        = 500;
CenterSph                   = [-2500 1000 rSph];
[Theta, Phi]                = meshgrid(linspace(-pi/2,pi/2,5), linspace(0,2*pi,10));
xSph                        = rSph * cos(Theta(:)) .* cos(Phi(:)) + CenterSph(1);
ySph                        = rSph * cos(Theta(:)) .* sin(Phi(:)) + CenterSph(2);
zSph                        = rSph * sin(Theta(:)) + CenterSph(3);

%%  One Snapshot View

viewMat                     = [-37.5 30; 0 0; 0 90];

fhdl                        = gobjects(size(viewMat,1), 1);
figureName                  = {'(3D View)', '(X-Z View)', '(X-Y View)'};
figureLoc                   = {'center', 'west', 'east'};

for i = 1:size(viewMat,1)

    fhdl(i)                 = figure('Name',['Reference Image - ', figureName{i}],'NumberTitle','off',...
        'Position', [0 0 floor(ScreenSize(3)/3) floor(ScreenSize(4)/3)]);
    movegui(figureLoc{i})

    gHndl                   = gca;
    gHndl.NextPlot          = 'replacechildren';
    gHndl.XTick             = -3500:1000:3500;
    gHndl.YTick             = -3500:1000:3500;
    gHndl.ZTick             = 0:1000:2000;
    gHndl.XTickLabel        = mat2cell(gHndl.XTick / 1e3,1,length(gHndl.XTick));
    gHndl.YTickLabel        = mat2cell(gHndl.YTick / 1e3,1,length(gHndl.YTick));
    gHndl.ZTickLabel        = mat2cell(gHndl.ZTick / 1e3,1,length(gHndl.ZTick));
    axis(gHndl,'equal')
    axis(gHndl,[-3500 3500 -3500 3500 0 2000])
    grid(gHndl,'on')
    box(gHndl,'on')
    hold(gHndl,'on')
    view(gHndl,viewMat(i,:))
    xlabel(gHndl,'$x$ (m)','FontSize',11,'Interpreter','Latex')
    ylabel(gHndl,'$y$ (m)','FontSize',11,'Interpreter','Latex')
    zlabel(gHndl,'$z$ (m)','FontSize',11,'Interpreter','Latex')

    plot3(gHndl,dataFrame(frameNo,:,1),dataFrame(frameNo,:,3),dataFrame(frameNo,:,2), '.','MarkerSize',12)
    plot3(gHndl,xPyr,yPyr,zPyr, '.','MarkerSize',12)
    plot3(gHndl,xSph,ySph,zSph, '.','MarkerSize',12)

end