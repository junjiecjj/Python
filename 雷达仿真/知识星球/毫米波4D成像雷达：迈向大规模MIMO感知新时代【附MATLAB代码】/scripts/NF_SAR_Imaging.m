%%  PreProcessing
clear, close all
clc
%%  Initialization

addpath('../data/')
addpath(genpath('../functions/'))

%%  Antenna Aperture

% - No. of Transceiver Antennas
Ntrx                        = 256;
Ntrz                        = 256;

% - Array Aperture length (m)
Lx                          = 2;
Lz                          = 2;

% - Spatial and Spatial Frequency Domains
if Ntrx == 1
    xtr = 0;
    kxtr = 0;
else
    xtr = linspace(-Lx/2 , Lx/2 , Ntrx);                                    % Spatial Domain in x-axis
    kxtr = 2*pi*(0:Ntrx-1)/Ntrx/(xtr(2)-xtr(1));                            % spatial Frequency Domain in x-axis
end
if Ntrz == 1
    ztr = 0;
    kztr = 0;
else
    ztr = linspace(-Lz/2 , Lz/2 , Ntrz);                                    % Spatial Domain in z-axis
    kztr = 2*pi*(0:Ntrz-1)/Ntrz/(ztr(2)-ztr(1));                            % Spatial Frequency Domain in z-axis
end

%%  Target Space

% - Scaling Factor
DataScale                   = [4e3 4e3 4e3];

% - Scaling Target Space Dimension (m)
Dx                          = 1;
Dz                          = 1;
Dy                          = 1;

% - Antenna Aperture to Center of Scaled Target Space (m)
R0                          = 1;

% - Scatterers Points and Properties of Static Seen (Pyramid)
ScalePyr                    = 500;
BaseCenterPyr               = [2750, 1250, 0];
[xsPyr, ysPyr, zsPyr]       = PyramidSurf(4, ScalePyr, BaseCenterPyr);

xsPyr                       = xsPyr / DataScale(1);
ysPyr                       = ysPyr / DataScale(2);
zsPyr                       = zsPyr / DataScale(3);

NsPyr                       = length(xsPyr);
fsPyr                       = ones(NsPyr,1);

% - Scatterers Points and Properties of Static Seen (Sphere)
rSph                        = 500;
CenterSph                   = [-2500 1000 rSph];
[ThetaSph, PhiSph]          = meshgrid(linspace(-pi/2,pi/2,5), linspace(0,2*pi,10));
xsSph                       = (rSph * cos(ThetaSph(:)) .* cos(PhiSph(:)) + CenterSph(1)) / DataScale(1);
ysSph                       = (rSph * cos(ThetaSph(:)) .* sin(PhiSph(:)) + CenterSph(2)) / DataScale(2);
zsSph                       = (rSph * sin(ThetaSph(:)) + CenterSph(3)) / DataScale(3);

NsSph                       = length(xsSph);
fsSph                       = ones(NsSph,1);

% - Scatterers Points and Properties of Dynamic Seen (Man Walking in Circle)
load dataFrame
dataFrame                   = dataFrame ./ permute(DataScale, [1, 3, 2]);

dataFrameCyclic             = cat(1, dataFrame, flip(dataFrame, 1));
dataFrameCyclic             = repmat(dataFrameCyclic,ceil(Ntrx * Ntrz / size(dataFrameCyclic,1)), 1, 1);
dataFrameCyclic             = reshape(dataFrameCyclic(1:Ntrx * Ntrz, :, :), Ntrx, Ntrz, 1, size(dataFrameCyclic,2), size(dataFrameCyclic,3));

xsMot                       = dataFrameCyclic(:,:,:,:,1);
ysMot                       = dataFrameCyclic(:,:,:,:,3);
zsMot                       = dataFrameCyclic(:,:,:,:,2);

NsMot                       = size(dataFrameCyclic,4);
fsMot                       = ones(NsMot,1);

%%  Frequency Definitions

c                           = 3e8;
fmin                        = 9e9;                                              % Min. Operating Freq.
fmax                        = 15e9;                                             % Max. Operating Freq.
fstep                       = 3e7;                                              % Freq. Step (for Sampling)
fo                          = fmin:fstep:fmax;                                  % Frequncy Domain
k                           = 2*pi*fo/c;                                        % Wavenumber
Nk                          = numel(k);                                         % No. of Sampling Frequency

%%  Data Acquisition - Freq. Domain Data Sampling

% - Pyramid
SPyr                        = zeros(Ntrx,Ntrz,Nk);

xtrPPyr                     = xtr.';
ztrPPyr                     = ztr;
xsPPyr                      = permute(xsPyr,[4, 3, 2, 1]);
ysPPyr                      = permute(ysPyr,[4, 3, 2, 1]);
zsPPyr                      = permute(zsPyr,[4, 3, 2, 1]);
fsPPyr                      = permute(fsPyr,[4, 3, 2, 1]);
kPPyr                       = permute(k,[3, 1, 2]);

RtrPyr                      = sqrt((xtrPPyr-xsPPyr).^2 + (R0-ysPPyr).^2 + (ztrPPyr-zsPPyr).^2);

for kk = 1:NsPyr
    SPyr = SPyr + (exp(-1i*2*kPPyr .* RtrPyr(:,:,:,kk)).*exp(+1i*2*kPPyr*R0)).*fsPPyr(:,:,:,kk);
end

% - Sphere
SSph                        = zeros(Ntrx,Ntrz,Nk);

xtrPSph                     = xtr.';
ztrPSph                     = ztr;
xsPSph                      = permute(xsSph,[4, 3, 2, 1]);
ysPSph                      = permute(ysSph,[4, 3, 2, 1]);
zsPSph                      = permute(zsSph,[4, 3, 2, 1]);
fsPSph                      = permute(fsSph,[4, 3, 2, 1]);
kPSph                       = permute(k,[3, 1, 2]);

RtrSph                      = sqrt((xtrPSph-xsPSph).^2 + (R0-ysPSph).^2 + (ztrPSph-zsPSph).^2);

for kk = 1:NsSph
    SSph = SSph + (exp(-1i*2*kPSph .* RtrSph(:,:,:,kk)).*exp(+1i*2*kPSph*R0)).*fsPSph(:,:,:,kk);
end

% - Man Walking in Circle
SMot                        = zeros(Ntrx,Ntrz,Nk);

xtrPMot                     = xtr.';
ztrPMot                     = ztr;
fsPMot                      = permute(fsMot,[4, 3, 2, 1]);
kPMot                       = permute(k,[3, 1, 2]);

RtrMot                      = sqrt((xtrPMot-xsMot).^2 + (R0-ysMot).^2 + (ztrPMot-zsMot).^2);

for kk = 1:NsMot
    SMot = SMot + (exp(-1i*2*kPMot .* RtrMot(:,:,:,kk)).*exp(+1i*2*kPMot*R0)).*fsPMot(:,:,:,kk);
end

%%  2D Cross Range FFT

% - Pyramid
SfPyr                       = fft2(SPyr);

% - Sphere
SfSph                       = fft2(SSph);

% - Man Walking in Circle
SfMot                       = fft2(SMot);

%%  Matched Filtering

% - Pyramid
SmPyr                       = zeros(Ntrx,Ntrz,Nk);
for m = 1:Ntrx
    for n = 1:Ntrz
        for w = 1:Nk
            temp_ky = 4*k(w)^2-kxtr(m)^2-kztr(n)^2;
            if temp_ky > 0
                temp_s_ky = sqrt(temp_ky);
                SmPyr(m,n,w) = SfPyr(m,n,w)*2*pi*(1/(1i*temp_s_ky))*exp(-1i*(2*k(w)-temp_s_ky)*R0);
            end
        end
    end
end

% - Sphere
SmSph                       = zeros(Ntrx,Ntrz,Nk);
for m = 1:Ntrx
    for n = 1:Ntrz
        for w = 1:Nk
            temp_ky = 4*k(w)^2-kxtr(m)^2-kztr(n)^2;
            if temp_ky > 0
                temp_s_ky = sqrt(temp_ky);
                SmSph(m,n,w) = SfSph(m,n,w)*2*pi*(1/(1i*temp_s_ky))*exp(-1i*(2*k(w)-temp_s_ky)*R0);
            end
        end
    end
end

% - Man Walking in Circle
SmMot                       = zeros(Ntrx,Ntrz,Nk);
for m = 1:Ntrx
    for n = 1:Ntrz
        for w = 1:Nk
            temp_ky = 4*k(w)^2-kxtr(m)^2-kztr(n)^2;
            if temp_ky > 0
                temp_s_ky = sqrt(temp_ky);
                SmMot(m,n,w) = SfMot(m,n,w)*2*pi*(1/(1i*temp_s_ky))*exp(-1i*(2*k(w)-temp_s_ky)*R0);
            end
        end
    end
end

%%  Interpolation

ky                          = linspace(0,2*max(k), Nk);
fark                        = ky(2) - ky(1);
Ntry                        = numel(ky);
Nxx                         = 400;
Nzz                         = 400;
Nyy                         = Ntry;

% - Pyramid
SSPyr                       = zeros(Nxx,Nyy,Nzz);
for m = 1:Ntrx
    for n = 1:Ntrz
        for w = 1:Nk
            temp_ky = 4*k(w)^2-kxtr(m)^2-kztr(n)^2;
            if temp_ky > 0
                temp_s_ky = sqrt(temp_ky);
                index = round((temp_s_ky)/fark)+1;
                if index > 0
                    SSPyr(m,index,n) = SSPyr(m,index,n)+SmPyr(m,n,w);
                else
                end
            end
        end
    end
end

% - Sphere
SSSph                       = zeros(Nxx,Nyy,Nzz);
for m = 1:Ntrx
    for n = 1:Ntrz
        for w = 1:Nk
            temp_ky = 4*k(w)^2-kxtr(m)^2-kztr(n)^2;
            if temp_ky > 0
                temp_s_ky = sqrt(temp_ky);
                index = round((temp_s_ky)/fark)+1;
                if index > 0
                    SSSph(m,index,n) = SSSph(m,index,n)+SmSph(m,n,w);
                else
                end
            end
        end
    end
end

% - Man Walking in Circle
SSMot                       = zeros(Nxx,Nyy,Nzz);
for m = 1:Ntrx
    for n = 1:Ntrz
        for w = 1:Nk
            temp_ky = 4*k(w)^2-kxtr(m)^2-kztr(n)^2;
            if temp_ky > 0
                temp_s_ky = sqrt(temp_ky);
                index = round((temp_s_ky)/fark)+1;
                if index > 0
                    SSMot(m,index,n) = SSMot(m,index,n)+SmMot(m,n,w);
                else
                end
            end
        end
    end
end

%%  Show Reflectivity Image

% - Pyrmid
SsonPyr                     = flip(fftshift(ifftn(SSPyr), 2), 2); 
SsonPyr                     = abs(SsonPyr);
SimagePyr                   = SsonPyr/max(SsonPyr(:));

x_samplesPyr                = linspace(-Dx,Dx,Nxx);
z_samplesPyr                = linspace(-Dz,Dz,Nzz);
y_samplesPyr                = linspace(-Dy, Dy, Nyy);

% - Sphere
SsonSph                     = flip(fftshift(ifftn(SSSph), 2), 2); 
SsonSph                     = abs(SsonSph);
SimageSph                   = SsonSph/max(SsonSph(:));

x_samplesSph                = linspace(-Dx,Dx,Nxx);
z_samplesSph                = linspace(-Dz,Dz,Nzz);
y_samplesSph                = linspace(-Dy, Dy, Nyy);

% - Man Walking in Circle
SsonMot                     = flip(fftshift(ifftn(SSMot), 2), 2); 
SsonMot                     = abs(SsonMot);
SimageMot                   = SsonMot/max(SsonMot(:));

x_samplesMot                = linspace(-Dx,Dx,Nxx);
z_samplesMot                = linspace(-Dz,Dz,Nzz);
y_samplesMot                = linspace(-Dy, Dy, Nyy);

%%  Data Processing

% - Pyramid
Simage_dbPyr                = 20*log10(SimagePyr);
Simage_db_sortPyr           = sort(Simage_dbPyr(:),'descend');

fsparsePyr                  = Simage_dbPyr > Simage_db_sortPyr(floor(1e-4*length(Simage_db_sortPyr)));
[XPyr, YPyr, ZPyr]          = ndgrid(x_samplesPyr,y_samplesPyr,z_samplesPyr);

XplotPyr                    = XPyr(fsparsePyr) * DataScale(1);
YplotPyr                    = YPyr(fsparsePyr) * DataScale(2);
ZplotPyr                    = ZPyr(fsparsePyr) * DataScale(3);

% - Sphere
Simage_dbSph                = 20*log10(SimageSph);
Simage_db_sortSph           = sort(Simage_dbSph(:),'descend');

fsparseSph                  = Simage_dbSph > Simage_db_sortSph(floor(1e-4*length(Simage_db_sortSph)));
[XSph, YSph, ZSph]          = ndgrid(x_samplesSph,y_samplesSph,z_samplesSph);

XplotSph                    = XSph(fsparseSph) * DataScale(1);
YplotSph                    = YSph(fsparseSph) * DataScale(2);
ZplotSph                    = ZSph(fsparseSph) * DataScale(3);

% - Man Walking in Circle
Simage_dbMot                = 20*log10(SimageMot);
Simage_db_sortMot           = sort(Simage_dbMot(:),'descend');

fsparseMot                  = Simage_dbMot > Simage_db_sortMot(floor(1e-4*length(Simage_db_sortMot)));
[XMot, YMot, ZMot]          = ndgrid(x_samplesMot,y_samplesMot,z_samplesMot);

XplotMot                    = XMot(fsparseMot) * DataScale(1);
YplotMot                    = YMot(fsparseMot) * DataScale(2);
ZplotMot                    = ZMot(fsparseMot) * DataScale(3);

%%  Plot Results

ScreenSize                  = get(0,'ScreenSize');

% - Initialization (3D View)
fhdl(1)                     = figure('Name','SAR Image - (3D View)','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/3) floor(ScreenSize(4)/3)]);
movegui('center')

gHndl                       = gca;
gHndl.NextPlot              = 'replacechildren';
gHndl.XTick                 = -3500:1000:3500;
gHndl.YTick                 = -3500:1000:3500;
gHndl.ZTick                 = -2000:1000:2000;
gHndl.XTickLabel            = mat2cell(gHndl.XTick / 1e3,1,length(gHndl.XTick));
gHndl.YTickLabel            = mat2cell(gHndl.YTick / 1e3,1,length(gHndl.YTick));
gHndl.ZTickLabel            = mat2cell(gHndl.ZTick / 1e3,1,length(gHndl.ZTick));
axis(gHndl,'equal')
axis(gHndl,[-3500 3500 -3500 3500 0 2000])
grid(gHndl,'on')
box(gHndl,'on')
hold(gHndl,'on')
view(gHndl,-37.5,30)
xlabel(gHndl, '$x$ (m)','FontSize',11,'Interpreter','Latex')
ylabel(gHndl, '$y$ (m)','FontSize',11,'Interpreter','Latex')
zlabel(gHndl, '$z$ (m)','FontSize',11,'Interpreter','Latex')

% - Pyramid
plot3(XplotPyr(:), YplotPyr(:), ZplotPyr(:), '.','MarkerSize',12, 'Color', [0.8500, 0.3250, 0.0980])

% - Sphere
plot3(XplotSph(:), YplotSph(:), ZplotSph(:), '.','MarkerSize',12, 'Color', [0.9290 0.6940 0.1250])

% - Man Walking in Circle
plot3(XplotMot(:),YplotMot(:),ZplotMot(:), '.','MarkerSize',12, 'Color', [0 0.4470 0.7410])

% - Initialization (X-Z View)
fhdl(2)                     = figure('Name','SAR Image - (X-Z View)','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/3) floor(ScreenSize(4)/3)]);
movegui('east')

gHndl                       = gca;
gHndl.NextPlot              = 'replacechildren';
gHndl.XTick                 = -3500:1000:3500;
gHndl.YTick                 = -3500:1000:3500;
gHndl.ZTick                 = -2000:1000:2000;
gHndl.XTickLabel            = mat2cell(gHndl.XTick / 1e3,1,length(gHndl.XTick));
gHndl.YTickLabel            = mat2cell(gHndl.YTick / 1e3,1,length(gHndl.YTick));
gHndl.ZTickLabel            = mat2cell(gHndl.ZTick / 1e3,1,length(gHndl.ZTick));
axis(gHndl,'equal')
axis(gHndl,[-3500 3500 -3500 3500 0 2000])
grid(gHndl,'on')
box(gHndl,'on')
hold(gHndl,'on')
view(gHndl,0,0)
xlabel(gHndl, '$x$ (m)','FontSize',11,'Interpreter','Latex')
ylabel(gHndl, '$y$ (m)','FontSize',11,'Interpreter','Latex')
zlabel(gHndl, '$z$ (m)','FontSize',11,'Interpreter','Latex')

% - Pyramid
plot3(XplotPyr(:), YplotPyr(:), ZplotPyr(:), '.','MarkerSize',12, 'Color', [0.8500, 0.3250, 0.0980])

% - Sphere
plot3(XplotSph(:), YplotSph(:), ZplotSph(:), '.','MarkerSize',12, 'Color', [0.9290 0.6940 0.1250])

% - Man Walking in Circle
plot3(XplotMot(:),YplotMot(:),ZplotMot(:), '.','MarkerSize',12, 'Color', [0 0.4470 0.7410])

% - Initialization (X-Y View)
fhdl(3)                     = figure('Name','SAR Image - (X-Y View)','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/3) floor(ScreenSize(4)/3)]);
movegui('west')

gHndl                       = gca;
gHndl.NextPlot              = 'replacechildren';
gHndl.XTick                 = -3500:1000:3500;
gHndl.YTick                 = -3500:1000:3500;
gHndl.ZTick                 = -2000:1000:2000;
gHndl.XTickLabel            = mat2cell(gHndl.XTick / 1e3,1,length(gHndl.XTick));
gHndl.YTickLabel            = mat2cell(gHndl.YTick / 1e3,1,length(gHndl.YTick));
gHndl.ZTickLabel            = mat2cell(gHndl.ZTick / 1e3,1,length(gHndl.ZTick));
axis(gHndl,'equal')
axis(gHndl,[-3500 3500 -3500 3500 0 2000])
grid(gHndl,'on')
box(gHndl,'on')
hold(gHndl,'on')
view(gHndl,0,90)
xlabel(gHndl, '$x$ (m)','FontSize',11,'Interpreter','Latex')
ylabel(gHndl, '$y$ (m)','FontSize',11,'Interpreter','Latex')
zlabel(gHndl, '$z$ (m)','FontSize',11,'Interpreter','Latex')

% - Pyramid
plot3(XplotPyr(:), YplotPyr(:), ZplotPyr(:), '.','MarkerSize',12, 'Color', [0.8500, 0.3250, 0.0980])

% - Sphere
plot3(XplotSph(:), YplotSph(:), ZplotSph(:), '.','MarkerSize',12, 'Color', [0.9290 0.6940 0.1250])

% - Man Walking in Circle
plot3(XplotMot(:),YplotMot(:),ZplotMot(:), '.','MarkerSize',12, 'Color', [0 0.4470 0.7410])