% close all;
clear;clc
addpath('..\')
addpath('..\Function\')
Para = ParaClass();
%% Data Pre-processing
%% |    B_640_T_512
% if the target is not hit, then we don't show it in the CDF
index = [1 4 7];% -28 -26 -24
% SNR=-28dB
load('..\Data\B_640_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
All_RangeError_B_640_T_512_1 = rng_error(index(1),:)/( 4*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_640_T_512_1 = vel_error(index(1),:)...
    /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
All_AziError_B_640_T_512_1 = azi_error(index(1),:);

% delete the data with 0 error,since this indicates it is not hit
index_Range = find(All_RangeError_B_640_T_512_1 ~= 0);
index_Vel   = find(All_VelError_B_640_T_512_1 ~= 0);
index_Azi   = find(All_AziError_B_640_T_512_1 ~= 0);

All_RangeError_B_640_T_512_1 = All_RangeError_B_640_T_512_1(index_Range);
All_VelError_B_640_T_512_1 = All_VelError_B_640_T_512_1(index_Vel);
All_AziError_B_640_T_512_1 = All_AziError_B_640_T_512_1(index_Azi);

% SNR=-26dB
load('..\Data\B_640_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
All_RangeError_B_640_T_512_2 = rng_error(index(2),:)/( 4*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_640_T_512_2 = vel_error(index(2),:)...
    /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
All_AziError_B_640_T_512_2 = azi_error(index(2),:);

% delete the data with 0 error,since this indicates it is not hit
index_Range = find(All_RangeError_B_640_T_512_2 ~= 0);
index_Vel   = find(All_VelError_B_640_T_512_2 ~= 0);
index_Azi   = find(All_AziError_B_640_T_512_2 ~= 0);

All_RangeError_B_640_T_512_2 = All_RangeError_B_640_T_512_2(index_Range);
All_VelError_B_640_T_512_2 = All_VelError_B_640_T_512_2(index_Vel);
All_AziError_B_640_T_512_2 = All_AziError_B_640_T_512_2(index_Azi);

% SNR=-26dB
load('..\Data\B_640_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
All_RangeError_B_640_T_512_3 = rng_error(index(3),:)/( 4*Para.fs*Para.TcEff/...
            (3e8 * 2*(Para.Tsen+Para.Tcom)));
All_VelError_B_640_T_512_3 = vel_error(index(3),:)...
    /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
All_AziError_B_640_T_512_3 = azi_error(index(3),:);

% delete the data with 0 error,since this indicates it is not hit
index_Range = find(All_RangeError_B_640_T_512_3 ~= 0);
index_Vel   = find(All_VelError_B_640_T_512_3 ~= 0);
index_Azi   = find(All_AziError_B_640_T_512_3 ~= 0);

All_RangeError_B_640_T_512_3 = All_RangeError_B_640_T_512_3(index_Range);
All_VelError_B_640_T_512_3 = All_VelError_B_640_T_512_3(index_Vel);
All_AziError_B_640_T_512_3 = All_AziError_B_640_T_512_3(index_Azi);

% SNR=-26dB
% load('..\Data\B_640_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
% All_RangeError_B_640_T_512_4 = rng_error(index(4),:)/( 4*Para.fs*Para.TcEff/...
%             (3e8 * 2*(Para.Tsen+Para.Tcom)));
% All_VelError_B_640_T_512_4 = vel_error(index(4),:)...
%     /(2*Para.N_c*Para.TcAll*Para.fc/3e8);
% All_AziError_B_640_T_512_4 = azi_error(index(4),:);
% 
% % delete the data with 0 error,since this indicates it is not hit
% index_Range = find(All_RangeError_B_640_T_512_4 ~= 0);
% index_Vel   = find(All_VelError_B_640_T_512_4 ~= 0);
% index_Azi   = find(All_AziError_B_640_T_512_4 ~= 0);
% 
% All_RangeError_B_640_T_512_4 = All_RangeError_B_640_T_512_4(index_Range);
% All_VelError_B_640_T_512_4 = All_VelError_B_640_T_512_4(index_Vel);
% All_AziError_B_640_T_512_4 = All_AziError_B_640_T_512_4(index_Azi);
%%
set(0,'defaultfigurecolor','w') 
figure; hold on; box on; grid on;
%range
subplot(3,1,1); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(abs(All_RangeError_B_640_T_512_1));
    set(p1,'LineStyle','--', 'Color', 'r', 'LineWidth', 2) 
p2  = cdfplot(abs(All_RangeError_B_640_T_512_2));
    set(p2,'LineStyle','--', 'Color', 'g', 'LineWidth', 2) 
p3  = cdfplot(abs(All_RangeError_B_640_T_512_3));
    set(p3,'LineStyle','--', 'Color', 'b', 'LineWidth', 2) 
% p4  = cdfplot(abs(All_RangeError_B_640_T_512_4));
%     set(p4,'LineStyle','--', 'Color', 'c', 'LineWidth', 2)     
xlim([1e-3 3e8/2/Para.B*(Para.TcEff+Para.Tcom*4)/Para.TcEff])
xlabel('Distance Error [m]');
ylabel('CDF');
title('')
set(gca, 'XScale', 'log');    
l1.FontSize = 12;
l1.FontName = 'Times New Roman';

%velocity
subplot(3,1,2); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(abs(All_VelError_B_640_T_512_1));
    set(p1,'LineStyle','--', 'Color', 'r', 'LineWidth', 2) 
p2  = cdfplot(abs(All_VelError_B_640_T_512_2));
    set(p2,'LineStyle','--', 'Color', 'g', 'LineWidth', 2) 
p3  = cdfplot(abs(All_VelError_B_640_T_512_3));
    set(p3,'LineStyle','--', 'Color', 'b', 'LineWidth', 2) 
% p4  = cdfplot(abs(All_VelError_B_640_T_512_4));
%     set(p4,'LineStyle','--', 'Color', 'c', 'LineWidth', 2)   
xlim([1e-3 3e8/(2*Para.N_c*Para.TcAll*Para.fc)])
xlabel('Velocity Error [m/s]');
ylabel('CDF');
title('')
set(gca, 'XScale', 'log');    
l1.FontSize = 12;
l1.FontName = 'Times New Roman';

%azimuth
subplot(3,1,3); hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(abs(All_AziError_B_640_T_512_1));
    set(p1,'LineStyle','--', 'Color', 'r', 'LineWidth', 2) 
p2  = cdfplot(abs(All_AziError_B_640_T_512_2));
    set(p2,'LineStyle','--', 'Color', 'g', 'LineWidth', 2) 
p3  = cdfplot(abs(All_AziError_B_640_T_512_3));
    set(p3,'LineStyle','--', 'Color', 'b', 'LineWidth', 2) 
% p4  = cdfplot(abs(All_AziError_B_640_T_512_4));
%     set(p4,'LineStyle','--', 'Color', 'c', 'LineWidth', 2)     
xlabel('Azimuth Error [deg]');
ylabel('CDF');
title('') 
ah1=axes('position',get(gca,'position'),'visible','off'); 
l1 = legend([p1, p2, p3],{  
        'SNR = -25 dB',...
        'SNR = -38 dB',...
        'SNR = -31 dB'},...
        'Interpreter','tex', 'Box','off',...
        'NumColumns', 2);
l1.FontSize = 12;
l1.FontName = 'Times New Roman';