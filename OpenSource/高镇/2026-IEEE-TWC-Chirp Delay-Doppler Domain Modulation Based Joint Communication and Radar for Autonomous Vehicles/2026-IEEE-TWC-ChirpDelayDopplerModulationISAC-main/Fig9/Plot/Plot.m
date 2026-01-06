IM = load('IM.mat');
QAM = load('QAM.mat');
IMQAM = load('IMQAM.mat');
AntennaNum = 4;
N_f = 1024;
N_c = 128;
%% DDM
set(0,'defaultfigurecolor','w')
figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
% IMQAM_16
p1 = plot(IMQAM.SNR-floor(10*log10(AntennaNum*N_f*N_c)), mean(squeeze(IMQAM.DDM_data(2,:,:)),1), 'LineStyle', '-', 'Marker', 's', 'Markersize', 10, 'Color', 'r', 'LineWidth', 3);
% IM_16
p2 = plot(IMQAM.SNR-floor(10*log10(AntennaNum*N_f*N_c)), mean(squeeze(IM.DDM_data(2,:,:)),1), 'LineStyle', '-', 'Marker', 'o', 'Markersize', 10, 'Color', 'r', 'LineWidth', 3);
% QAM_16
p3 = plot(IMQAM.SNR-floor(10*log10(AntennaNum*N_f*N_c)), mean(squeeze(QAM.DDM_data(2,:,:)),1), 'LineStyle', '-', 'Marker', '+', 'Markersize', 10, 'Color', 'r', 'LineWidth', 3);
% IMQAM_4
p4 = plot(IMQAM.SNR-floor(10*log10(AntennaNum*N_f*N_c)), mean(squeeze(IMQAM.DDM_data(1,:,:)),1), 'LineStyle', '-.', 'Marker', '>', 'Markersize', 10, 'Color', 'b', 'LineWidth', 3);
% IMQAM_64
p5 = plot(IMQAM.SNR-floor(10*log10(AntennaNum*N_f*N_c)), mean(squeeze(IMQAM.DDM_data(3,:,:)),1), 'LineStyle', '-.', 'Marker', '<', 'Markersize', 10, 'Color', 'g', 'LineWidth', 3);

xlabel('SNR [dB]');
ylabel('Achievable Rate [bits/symbol]');

legend([p1, p2, p3, p4, p5], {
    'DD-16QAM',...
    'DD',...
    '16QAM',...
    'DD-4QAM',...
    'DD-64QAM'},...
    'Interpreter', 'tex', 'NumColumns', 1, 'FontSize', 16, 'FontName', 'Times New Roman');

%% TDM
set(0,'defaultfigurecolor','w')
figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
% IMQAM_16
p1 = plot(IMQAM.SNR-floor(10*log10(N_f)), mean(squeeze(IMQAM.TDM_data(2,:,:)),1), 'LineStyle', '-', 'Marker', 's', 'Markersize', 10, 'Color', 'r', 'LineWidth', 3);
% IM_16
p2 = plot(IMQAM.SNR-floor(10*log10(N_f)), mean(squeeze(IM.TDM_data(2,:,:)),1), 'LineStyle', '-', 'Marker', 'o', 'Markersize', 10, 'Color', 'r', 'LineWidth', 3);
% QAM_16
p3 = plot(IMQAM.SNR-floor(10*log10(N_f)), mean(squeeze(QAM.TDM_data(2,:,:)),1), 'LineStyle', '-', 'Marker', '+', 'Markersize', 10, 'Color', 'r', 'LineWidth', 3);
% IMQAM_4
p4 = plot(IMQAM.SNR-floor(10*log10(N_f)), mean(squeeze(IMQAM.TDM_data(1,:,:)),1), 'LineStyle', '-.', 'Marker', '>', 'Markersize', 10, 'Color', 'b', 'LineWidth', 3);
% IMQAM_64
p5 = plot(IMQAM.SNR-floor(10*log10(N_f)), mean(squeeze(IMQAM.TDM_data(3,:,:)),1), 'LineStyle', '-.', 'Marker', '<', 'Markersize', 10, 'Color', 'g', 'LineWidth', 3);

xlabel('SNR [dB]');
ylabel('Achievable Rate [bits/symbol]');

% legend([p1, p2, p3, p4, p5], {
%     'DD-16QAM',...
%     'DD',...
%     '16QAM',...
%     'DD-4QAM',...
%     'DD-64QAM'},...
%     'Interpreter', 'tex', 'NumColumns', 1, 'FontSize', 16, 'FontName', 'Times New Roman');