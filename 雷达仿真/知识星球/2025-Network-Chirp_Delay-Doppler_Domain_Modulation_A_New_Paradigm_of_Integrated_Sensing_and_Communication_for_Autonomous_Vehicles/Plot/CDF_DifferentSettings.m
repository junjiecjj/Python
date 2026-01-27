% close all;
clear;clc
addpath('..\')
addpath('..\Function\')
Para = ParaClass_640_512();
%% Data Pre-processing
%% |    B_640_T_512
% if the target is not hit, then we don't show it in the CDF
load('..\Data\B_640_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
All_RangeError_B_640_T_512 = rng_error(1,:)/( 4*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_640_T_512 = vel_error(1,:)...
    /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
All_AziError_B_640_T_512 = azi_error(1,:);

% delete the data with 0 error,since this indicates it is not hit
index_Range = find(All_RangeError_B_640_T_512 ~= 0);
index_Vel   = find(All_VelError_B_640_T_512 ~= 0);
index_Azi   = find(All_AziError_B_640_T_512 ~= 0);

All_RangeError_B_640_T_512 = All_RangeError_B_640_T_512(index_Range);
All_VelError_B_640_T_512 = All_VelError_B_640_T_512(index_Vel);
All_AziError_B_640_T_512 = All_AziError_B_640_T_512(index_Azi);
%% |    B_320_T_512
Para = ParaClass_320_512();
load('..\Data\B_320_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
All_RangeError_B_320_T_512 = rng_error(1,:)/( 4*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_320_T_512 = vel_error(1,:)...
    /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
All_AziError_B_320_T_512 = azi_error(1,:);

% delete the data with 0 error,since this indicates it is not hit
index_Range = find(All_RangeError_B_320_T_512 ~= 0);
index_Vel   = find(All_VelError_B_320_T_512 ~= 0);
index_Azi   = find(All_AziError_B_320_T_512 ~= 0);

All_RangeError_B_320_T_512 = All_RangeError_B_320_T_512(index_Range);
All_VelError_B_320_T_512 = All_VelError_B_320_T_512(index_Vel);
All_AziError_B_320_T_512 = All_AziError_B_320_T_512(index_Azi);
%% |    B_160_T_512
Para = ParaClass_160_512();
load('..\Data\B_160_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
All_RangeError_B_160_T_512 = rng_error(1,:)/( 4*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_160_T_512 = vel_error(1,:)...
    /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
All_AziError_B_160_T_512 = azi_error(1,:);
% delete the data with 0 error,since this indicates it is not hit
index_Range = find(All_RangeError_B_160_T_512 ~= 0);
index_Vel   = find(All_VelError_B_160_T_512 ~= 0);
index_Azi   = find(All_AziError_B_160_T_512 ~= 0);

All_RangeError_B_160_T_512 = All_RangeError_B_160_T_512(index_Range);
All_VelError_B_160_T_512 = All_VelError_B_160_T_512(index_Vel);
All_AziError_B_160_T_512 = All_AziError_B_160_T_512(index_Azi);
%% |    B_640_T_256
Para = ParaClass_640_256();
load('..\Data\B_640_T_256_SNR_25_35_Nit_1e4_rng_666.mat')
All_RangeError_B_640_T_256 = rng_error(1,:)/( 4*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_640_T_256 = vel_error(1,:)...
    /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
All_AziError_B_640_T_256 = azi_error(1,:);
% delete the data with 0 error,since this indicates it is not hit
index_Range = find(All_RangeError_B_640_T_256 ~= 0);
index_Vel   = find(All_VelError_B_640_T_256 ~= 0);
index_Azi   = find(All_AziError_B_640_T_256 ~= 0);

All_RangeError_B_640_T_256 = All_RangeError_B_640_T_256(index_Range);
All_VelError_B_640_T_256 = All_VelError_B_640_T_256(index_Vel);
All_AziError_B_640_T_256 = All_AziError_B_640_T_256(index_Azi);
%%
set(0,'defaultfigurecolor','w') 
figure; hold on; box on; grid on;
%range
subplot(3,1,1); hold on; box on; grid on;
xlim([1e-3 1e0])
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(abs(All_RangeError_B_640_T_512));
    set(p1,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p2  = cdfplot(abs(All_RangeError_B_320_T_512));
    set(p2,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p3  = cdfplot(abs(All_RangeError_B_160_T_512));
    set(p3,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p4  = cdfplot(abs(All_RangeError_B_640_T_256));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)    
xlabel('Distance Error [m]');
ylabel('CDF');
title('')
set(gca, 'XScale', 'log');    
l1.FontSize = 12;
l1.FontName = 'Times New Roman';

%velocity
subplot(3,1,2); hold on; box on; grid on;
xlim([1e-4 1e0])
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(abs(All_VelError_B_640_T_512));
    set(p1,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p2  = cdfplot(abs(All_VelError_B_320_T_512));
    set(p2,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p3  = cdfplot(abs(All_VelError_B_160_T_512));
    set(p3,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p4  = cdfplot(abs(All_VelError_B_640_T_256));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)      
xlabel('Velocity Error [m/s]');
ylabel('CDF');
title('')
set(gca, 'XScale', 'log');    
l1.FontSize = 12;
l1.FontName = 'Times New Roman';

%azimuth
subplot(3,1,3); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(abs(All_AziError_B_640_T_512));
    set(p1,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p2  = cdfplot(abs(All_AziError_B_320_T_512));
    set(p2,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p3  = cdfplot(abs(All_AziError_B_160_T_512));
    set(p3,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p4  = cdfplot(abs(All_AziError_B_640_T_256));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)     
xlabel('Azimuth Error [deg]');
ylabel('CDF');
title('') 
ah1=axes('position',get(gca,'position'),'visible','off'); 
l1 = legend([p1, p2, p3, p4],{  
        'B=640 M, T_c=51.2 \mus',...
        'B=320 M, T_c=51.2 \mus',...
        'B=160 M, T_c=51.2 \mus',...
        'B=640 M, T_c=25.6 \mus'},...
        'Interpreter','tex', 'Box','off',...
        'NumColumns', 2);
l1.FontSize = 12;
l1.FontName = 'Times New Roman';