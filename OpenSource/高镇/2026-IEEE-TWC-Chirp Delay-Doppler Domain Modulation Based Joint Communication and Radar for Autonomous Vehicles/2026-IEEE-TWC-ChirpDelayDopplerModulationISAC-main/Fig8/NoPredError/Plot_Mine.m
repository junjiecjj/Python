%% Plot_Mine.m  ——  重绘 Mine.json 的 BER / SER 曲线
clear; clc; close all

res  = jsondecode( fileread("Mine.json") );
snrs = -30:5:-5;

% ---------- BER ----------
ber_all  = res.case1_both.BER;
ber_tone = res.case2_tone_only.BER;
ber_qpsk = res.case3_qpsk_only.BER;

set(0,'defaultfigurecolor','w');
figure; hold on; legend on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');
semilogy(snrs, ber_all,  'o-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Total BER');
semilogy(snrs, ber_tone, 'x-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Delay-only BER');
semilogy(snrs, ber_qpsk, 's-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','QPSK-only BER');
xlabel('SNR [dB]'); ylabel('BER');

% ---------- SER ----------
ser_all  = res.case1_both.SER;
ser_tone = res.case2_tone_only.SER_tone;
ser_qpsk = res.case3_qpsk_only.SER_qpsk;

figure; hold on; grid on; legend on;
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');
semilogy(snrs, ser_all,  'o-', 'MarkerSize', 8,'LineWidth', 2, 'DisplayName','Total SER');
semilogy(snrs, ser_tone, 'x-', 'MarkerSize', 8,'LineWidth', 2, 'DisplayName','Delay-only SER');
semilogy(snrs, ser_qpsk, 's-', 'MarkerSize', 8,'LineWidth', 2, 'DisplayName','QPSK-only SER');
xlabel('SNR [dB]'); ylabel('SER');
