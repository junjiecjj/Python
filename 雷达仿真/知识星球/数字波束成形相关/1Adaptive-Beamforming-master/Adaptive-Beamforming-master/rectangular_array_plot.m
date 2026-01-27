% Robert Baummer
% Calculates 3D antenna pattern for set of weights in spherical coordinates,
% converts it to rectangular coordinates and plots the antenna pattern.
function rectangular_array_plot(w)

global L;
global M;
global dx;
global dy;
global nI;
global AOA_i;
global AOA_s;

if nargin < 1
    w = ones(M,L);
end

%Control variable for manual beamsteering
manual_beamsteer = false;
%Control variable for plotting interferers
plot_interferers = true;

%If manually beamsteering set Bx, By
%ie if function is run directly to generate plots at fixed angles
if manual_beamsteer == true
    L = 9;
    M = 9;
    %Beamsteer angles [theta phi]
    AOAs = [pi/4 pi/4];%[pi/8 pi/4];
    %Phase delays for manual beamsteering
    Bx = -2*pi*dx*sin(AOAs(1))*cos(AOAs(2));
    By = -2*pi*dx*sin(AOAs(1))*sin(AOAs(2));
%for plotting adaptive arrays
else
    Bx = 0;
    By = 0;
end

%Set X,Y indices of array
index_x = (-floor(L/2):floor(L/2))';
index_y = (-floor(M/2):floor(M/2))';
%theta is 90 degress minus elevation angle (angle from positive z-axis to xy-plane)
%phi is angle counterclockwise from x-axis in xy-plane
%[theta phi] = meshgrid(-pi/2:pi/72:pi/2, -pi/2:pi/72:pi/2);
[theta phi] = meshgrid(-pi/2:pi/128:pi/2, -pi:pi/128:pi);
%Preallocate Array Factor
AF = zeros(size(theta));
%Calculate the sum of AFx*AFy
for i = 1:L
    for j = 1:M
        AFs = w(j,i)*exp(1i*(index_x(i)*(2*pi*dx*sin(theta).*cos(phi) + Bx) + index_y(j)*(2*pi*dy*sin(theta).*sin(phi) + By)));
        AF = AF + AFs;
    end
end

%Convert to cartesian coordinates
%Note angles are defined differently from AF equations
%function sph2cart(theta,phi,r)
%phi is elevation angle from xy-plane (AF 90-theta)
%theta is angle counterclockwise from x-axis in xy-plane (AF phi)
[x,y,z] = sph2cart(phi,pi/2-theta,abs(AF));

figure
title('Rectangular Array Antenna Pattern');
mesh(x, y, z);
%colormap([0 0 0]);
xlabel('x');
ylabel('y');
zlabel('z');
axis equal

%Plot interferers as red lines
if plot_interferers == true
    hold on;
    [x,y,z] = sph2cart(AOA_s(2),pi/2-AOA_s(1),[0 1.25]);
    plot3(x,y,z,'v-b');
    for i = 1:nI
        [x,y,z] = sph2cart(AOA_i(i,2),pi/2-AOA_i(i,1),[0 .5]);
        plot3(x,y,z,'v-r');
    end
end