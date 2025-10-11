clc;
close all;
clear all;

%%
S0 = [-5, 0]*10^3;
S1 = [5, 0]*10^3;
sigma_angle = 3*10^-3; 
sigma_S = 5; 
sigma_t = 20*10^-9;
%X = [1000, 5000];
%doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t)
N = 100; M = 60*10^3;
x = linspace(-M, M, N); y = linspace(-M, M, N);
gdop = zeros(N);
for i = 1:N
    for j = 1:N
        X(1) = x(i); X(2) = y(j);
        gdop(j,i) = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t)/1000;
    end
end

figure(1)
% subplot(1,2,1)
[c, h] = contour(x/1000, y/1000, gdop);
set(h,'ShowText','on','LevelList',[0:0.5:10]);
xlabel('x/km');
ylabel('y/km');
hold on;
title('sigma_a=3mrad,sigma_S=5m,sigma_t=20ns,value of GDOP/k');


%% 
S0 = [-5, 0]*10^3;
S1 = [5, 0]*10^3;
sigma_angle = 0; 
sigma_S = 0; 
sigma_t = 20*10^-9;
N = 100; M = 60*10^3;
x = linspace(-M, M, N); y = linspace(-M, M, N);
gdop = zeros(N);
for i = 1:N
    for j = 1:N
        X(1) = x(i); X(2) = y(j);
        gdop(j,i) = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t)/1000;
    end
end
figure(2)
% subplot(1,3,1)
[c, h] = contour(x/1000, y/1000, gdop);
set(h,'ShowText','on','LevelList',[0:0.1:1]);
xlabel('x/km');
ylabel('y/km');
hold on;


%%
S0 = [-5, 0]*10^3;
S1 = [5, 0]*10^3;
sigma_angle = 3*10^-3; 
sigma_S = 0; 
sigma_t = 0;
N = 100; M = 60*10^3;
x = linspace(-M, M, N); y = linspace(-M, M, N);
gdop = zeros(N);
for i = 1:N
    for j = 1:N
        X(1) = x(i); X(2) = y(j);
        gdop(j,i) = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t)/1000;
    end
end
figure(3)
% subplot(1,3,2)
[c, h] = contour(x/1000, y/1000, gdop);
set(h,'ShowText','on','LevelList',[0:0.5:5]);
xlabel('x/km');
ylabel('y/km');

%% 
S0 = [-5, 0]*10^3;
S1 = [5, 0]*10^3;
sigma_angle = 0; 
sigma_S = 5; 
sigma_t = 0;
N = 100; M = 60*10^3;
x = linspace(-M, M, N); y = linspace(-M, M, N);
gdop = zeros(N);
for i = 1:N
    for j = 1:N
        X(1) = x(i); X(2) = y(j);
        gdop(j,i) = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t)/1000;
    end
end
figure(4)
% subplot(1,3,3)
[c, h] = contour(x/1000, y/1000, gdop);
set(h,'ShowText','on','LevelList',[0:0.1:1]);
xlabel('x/km');
ylabel('y/km');


function [delta_t, tan0, tan1] = doa_tdoa_param(S0, S1, X)
    c = 3*10^8;
    r0 = sqrt((X(1) - S0(1))^2 + (X(2) - S0(2))^2);
    r1 = sqrt((X(1) - S1(1))^2 + (X(2) - S1(2))^2);
    delta_t = (r1 - r0)/c;
    tan0 = (X(2) - S0(2))/(X(1) - S0(1));
    tan1 = (X(2) - S1(2))/(X(1) - S1(1));
end


function gdop = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t)
    x = X(1); y = X(2); x0 = S0(1); y0 = S0(2); x1 = S1(1); y1 = S1(2);
    c = 3*10^8;
    C = zeros(2); U = zeros(2); W = zeros(2);
    sigma_r = c * sigma_t;
    r1 = sqrt((x - x1)^2 + (y - y1)^2);
    r0 = sqrt((x - x0)^2 + (y - y0)^2);
    sin0 = (y - y0)/r0; cos0 = (x - x0)/r0;
    C = [-sin0^2/(y - y0), cos0^2/(x - x0); (x - x1)/r1 - (x - x0)/r0, (y - y1)/r1 - (y - y0)/r0];
    U = [sin0^2/(y - y0), -cos0^2/(x - x0); (x - x0)/r0, (y - y0)/r0];
    W = [0, 0; (x - x1)/r1, (y - y1)/r1];
    Rv = [sigma_angle^2, 0; 0, sigma_r^2];
    Rs = [sigma_S^2, 0; 0, sigma_S^2];
    Pdx = pinv(C) * (Rv + U * Rs * U' + W * Rs * W') * pinv(C)';
    gdop = sqrt(Pdx(1,1) + Pdx(2,2));
end