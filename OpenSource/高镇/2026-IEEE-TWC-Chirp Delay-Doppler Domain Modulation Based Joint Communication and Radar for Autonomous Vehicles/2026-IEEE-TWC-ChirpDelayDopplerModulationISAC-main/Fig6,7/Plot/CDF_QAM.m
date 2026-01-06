% close all;
clear; clc
addpath('..\')
addpath('..\Function\')

%% Marker settings (for black-and-white printing)
N_mark  = 6;                  % Number of markers per CDF curve
markers = {'o','s','^','d'};  % Marker styles for each data series
mSize   = 10;                  % Marker size

%% Data Pre-processing for B_640_T_512
% Load data
load('..\Data\BER_Hitrate\B_640_T_512.mat')
% Create parameter object
Para = ParaClass_640_512();
% Compute normalized range error (in meters)
All_RangeError_B_640_T_512 = rng_error(1,:) ...
    / (4 * Para.fs * Para.TcEff / (3e8 * 2*(Para.Tsen+Para.Tcom)));
% Compute normalized velocity error (in m/s)
All_VelError_B_640_T_512 = vel_error(1,:) ...
    / (2 * Para.N_c * Para.TcAll * Para.fc / 3e8);
% Take azimuth error (in degrees)
All_AziError_B_640_T_512 = azi_error(1,:);
% Remove zero‐error entries (indicates missed detections)
index_Range = find(All_RangeError_B_640_T_512 ~= 0);
index_Vel   = find(All_VelError_B_640_T_512   ~= 0);
index_Azi   = find(All_AziError_B_640_T_512   ~= 0);
All_RangeError_B_640_T_512 = All_RangeError_B_640_T_512(index_Range);
All_VelError_B_640_T_512   = All_VelError_B_640_T_512(index_Vel);
All_AziError_B_640_T_512   = All_AziError_B_640_T_512(index_Azi);

%% Data Pre-processing for B_640_T_512_16QAM
load('..\Data\BER_Hitrate\B_640_T_512_16QAM.mat')
All_RangeError_B_640_T_512_16QAM = rng_error(1,:) ...
    / (4 * Para.fs * Para.TcEff / (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_640_T_512_16QAM = vel_error(1,:) ...
    / (2 * Para.N_c * Para.TcAll * Para.fc / 3e8);
All_AziError_B_640_T_512_16QAM = azi_error(1,:);
index_Range = find(All_RangeError_B_640_T_512_16QAM ~= 0);
index_Vel   = find(All_VelError_B_640_T_512_16QAM   ~= 0);
index_Azi   = find(All_AziError_B_640_T_512_16QAM   ~= 0);
All_RangeError_B_640_T_512_16QAM = All_RangeError_B_640_T_512_16QAM(index_Range);
All_VelError_B_640_T_512_16QAM   = All_VelError_B_640_T_512_16QAM(index_Vel);
All_AziError_B_640_T_512_16QAM   = All_AziError_B_640_T_512_16QAM(index_Azi);

%% Data Pre-processing for B_640_T_512_64QAM
load('..\Data\BER_Hitrate\B_640_T_512_64QAM.mat')
All_RangeError_B_640_T_512_64QAM = rng_error(1,:) ...
    / (4 * Para.fs * Para.TcEff / (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_640_T_512_64QAM = vel_error(1,:) ...
    / (2 * Para.N_c * Para.TcAll * Para.fc / 3e8);
All_AziError_B_640_T_512_64QAM = azi_error(1,:);
index_Range = find(All_RangeError_B_640_T_512_64QAM ~= 0);
index_Vel   = find(All_VelError_B_640_T_512_64QAM   ~= 0);
index_Azi   = find(All_AziError_B_640_T_512_64QAM   ~= 0);
All_RangeError_B_640_T_512_64QAM = All_RangeError_B_640_T_512_64QAM(index_Range);
All_VelError_B_640_T_512_64QAM   = All_VelError_B_640_T_512_64QAM(index_Vel);
All_AziError_B_640_T_512_64QAM   = All_AziError_B_640_T_512_64QAM(index_Azi);
%% Plot CDFs with markers
set(0,'defaultfigurecolor','w')
figure;

% 1) Distance Error CDF
subplot(3,1,1); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
p1 = cdfplot(abs(All_RangeError_B_640_T_512));
x1 = get(p1,'XData'); idx1 = round(linspace(1,length(x1),N_mark));
set(p1, 'LineStyle','-', 'Color','b', 'LineWidth',2, ...
        'Marker',markers{1}, 'MarkerIndices',idx1, 'MarkerSize',mSize);
p2 = cdfplot(abs(All_RangeError_B_640_T_512_16QAM));
x2 = get(p2,'XData'); idx2 = round(linspace(1,length(x2),N_mark));
set(p2, 'LineStyle','-', 'Color','r', 'LineWidth',2, ...
        'Marker',markers{2}, 'MarkerIndices',idx2, 'MarkerSize',mSize);
p3 = cdfplot(abs(All_RangeError_B_640_T_512_64QAM));
x3 = get(p3,'XData'); idx3 = round(linspace(1,length(x3),N_mark));
set(p3, 'LineStyle','-', 'Color','g', 'LineWidth',2, ...
        'Marker',markers{3}, 'MarkerIndices',idx3, 'MarkerSize',mSize);
xlabel('Distance Error [m]');
ylabel('CDF');
title('')
xlim([1e-4 1]);
set(gca,'XScale','log');

% 2) Velocity Error CDF
subplot(3,1,2); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
p1 = cdfplot(abs(All_VelError_B_640_T_512));
x1 = get(p1,'XData'); idx1 = round(linspace(1,length(x1),N_mark));
set(p1, 'LineStyle','-', 'Color','b', 'LineWidth',2, ...
        'Marker',markers{1}, 'MarkerIndices',idx1, 'MarkerSize',mSize);
p2 = cdfplot(abs(All_VelError_B_640_T_512_16QAM));
x2 = get(p2,'XData'); idx2 = round(linspace(1,length(x2),N_mark));
set(p2, 'LineStyle','-', 'Color','r', 'LineWidth',2, ...
        'Marker',markers{2}, 'MarkerIndices',idx2, 'MarkerSize',mSize);
p3 = cdfplot(abs(All_VelError_B_640_T_512_64QAM));
x3 = get(p3,'XData'); idx3 = round(linspace(1,length(x3),N_mark));
set(p3, 'LineStyle','-', 'Color','g', 'LineWidth',2, ...
        'Marker',markers{3}, 'MarkerIndices',idx3, 'MarkerSize',mSize);
xlabel('Velocity Error [m/s]');
ylabel('CDF');
title('')
xlim([1e-3 1]);
set(gca,'XScale','log');

% 3) Azimuth Error CDF
subplot(3,1,3); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',12);
p1 = cdfplot(abs(All_AziError_B_640_T_512));
x1 = get(p1,'XData'); idx1 = round(linspace(1,length(x1),N_mark));
set(p1, 'LineStyle','-', 'Color','b', 'LineWidth',2, ...
        'Marker',markers{1}, 'MarkerIndices',idx1, 'MarkerSize',mSize);
p2 = cdfplot(abs(All_AziError_B_640_T_512_16QAM));
x2 = get(p2,'XData'); idx2 = round(linspace(1,length(x2),N_mark));
set(p2, 'LineStyle','-', 'Color','r', 'LineWidth',2, ...
        'Marker',markers{2}, 'MarkerIndices',idx2, 'MarkerSize',mSize);
p3 = cdfplot(abs(All_AziError_B_640_T_512_64QAM));
x3 = get(p3,'XData'); idx3 = round(linspace(1,length(x3),N_mark));
set(p3, 'LineStyle','-', 'Color','g', 'LineWidth',2, ...
        'Marker',markers{3}, 'MarkerIndices',idx3, 'MarkerSize',mSize);
xlabel('Azimuth Error [deg]');
ylabel('CDF');
title('')
xlim([0 1]);

% Add a single legend for all subplots
ah = axes('position',get(gca,'position'),'visible','off');
legend(ah, [p1,p2,p3], {
    '4QAM', ...
    '16QAM', ...
    '64QAM'}, ...
    'Interpreter','tex', 'Box','off', ...
     'NumColumns',1, ...
    'FontName','Times New Roman', 'FontSize',12);
