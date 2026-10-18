clear; clc; close all;

%% Define parameters
H = 10000 / 1000; % Height in kilometers
beta = 30 * pi / 180; % Angle in radians
Yc = H * tan(beta) / 1000; % Calculate Yc in kilometers

%% Define coordinates and values
coords = [
    Yc-3, 13.2;
    Yc-1, 12.9;
    Yc+1, 12.6;
    Yc+3, 12.3;
    Yc+3, 11.7;
    Yc+1, 11.4;
    Yc-1, 11.1;
    Yc-3, 10.8;
    Yc-12.5, 12.75;
    Yc-10, 12.5;
    Yc-7.5, 12.25;
    Yc-7.5, 11.75;
    Yc-10, 11.5;
    Yc-12.5, 11.25;
    Yc-12.5, 11.75;
    Yc-12.5, 12.25;
    Yc-10, 12;
    Yc-5, 12;
    Yc, 12;
    Yc+5, 12
];

%% Plot the points
hold on;
for i = 1:size(coords, 1)
    plot(coords(i, 1), coords(i, 2), 'ok', 'LineWidth', 10);
end

%% Set plot limits and labels
xlim([Yc-20, Yc+20]);
ylim([10, 15]);
xlabel('Range', 'FontSize', 14);
ylabel('Azimuth', 'FontSize', 14);
grid on;
box off;
