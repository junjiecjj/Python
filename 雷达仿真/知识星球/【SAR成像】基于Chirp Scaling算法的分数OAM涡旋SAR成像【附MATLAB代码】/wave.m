%%
clc;clear;close all; 
%% Parameters
c = 3e8;f = 4e10;lambda = c / f ;
k = 2 * pi / lambda; 
l = 0;
N = 8;
a = 120*lambda;
cc = 50;
H = 10000;
X = linspace (-cc, cc,1024);Y = linspace (-cc, cc,1024);
%% meshgrid function
[x,y] = meshgrid(X,Y);[phi, theta, r] = cart2sph(x,y,H);theta = pi/2 - theta;
%% votex wave
E = exp(-1i * k * r)./r.* N .* exp(1i * l * phi).*exp( 1i* pi/2).*besselj(l,k*a*sin(theta')); 
amplitude = E.*conj(E);
phase = angle(E)/pi*180;
%% Plot
figure(1);surf(x,y,phase);shading interp;
xlim([-50 50]);ylim([-50 50]);xticks(-50:25:50);yticks(-50:25:50)
%title('Phase ( \it{l} = 0 )');
view(2);colormap jet;colorbar();axis on
%%
figure(2);surf(x,y,amplitude);shading interp;
xlim([-50 50]);ylim([-50 50]);xticks(-50:25:50);yticks(-50:25:50)
%title('Amplitude ( \it{l} = 0 )');
colormap jet;axis on
