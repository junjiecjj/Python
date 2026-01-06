% close all;
clear;clc

figure
set(0,'defaultfigurecolor','w');
set(gca,'FontName','Times New Roman','FontSize',16);
box on; grid on;hold on; xlabel('SNR/dB');ylabel('Hitrate')
load('..\Data\BER_Hitrate\B_640_T_512.mat')
SNRdB_ = -35:-25;

RangeVelocityAngle = [HitRate.RangeVelocityAngle];
% RangeVelocityAngle = [Nit RangeVelocityAngle(1:end-1)];
p1 = plot(SNRdB_, fliplr(RangeVelocityAngle) / (Nit), ...
    'b->', 'LineWidth', 2, 'MarkerSize', 10);

load('..\Data\BER_Hitrate\B_640_T_256.mat')
RangeVelocityAngle = [HitRate.RangeVelocityAngle];
% RangeVelocityAngle = [Nit RangeVelocityAngle(1:end-1)];
p2 = plot(SNRdB_, fliplr(RangeVelocityAngle) / (Nit), ...
    'r-o', 'LineWidth', 2, 'MarkerSize', 10);

load('..\Data\BER_Hitrate\B_320_T_512.mat')
RangeVelocityAngle = [HitRate.RangeVelocityAngle];
% RangeVelocityAngle = [Nit RangeVelocityAngle(1:end-1)];
p3 = plot(SNRdB_, fliplr(RangeVelocityAngle) / (Nit), ...
    'g--<', 'LineWidth', 2, 'MarkerSize', 10);

load('..\Data\BER_Hitrate\B_160_T_512.mat')
RangeVelocityAngle = [HitRate.RangeVelocityAngle];
p4 = plot(SNRdB_, fliplr(RangeVelocityAngle) / (Nit), ...
    'c--<', 'LineWidth', 2, 'MarkerSize', 10);

ah1=axes('position',get(gca,'position'),'visible','off'); 
l1 = legend(ah1, [p1, p2, p3, p4],...
    {'B=640 MHz, T_c=51.2 \mus',...
     'B=640 MHz, T_c=25.6 \mus', ...
     'B=320 MHz, T_c=51.2 \mus', ...
     'B=160 MHz, T_c=51.2 \mus'},...
    'Box','off', ...
    'Interpreter','tex');
l1.FontSize = 16;   
l1.FontName = 'Times New Roman';