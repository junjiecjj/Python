%% Plot_FRaC.m  ——  重绘 FRaC.json 的 BER / SER 曲线
% 需要 MATLAB R2016b+（支持 jsondecode）

clear; clc; close all

% ① 读取 JSON
res  = jsondecode( fileread("FRaC.json") );

% ② SNR 轴
snrs = -30:5:-5;

% ③ 配置
modes   = {'spatial','qpsk','all'};
markers = struct('spatial','s-','qpsk','^-','all','o-');
marker_perm = 'd-';

set(0,'defaultfigurecolor','w');
% ---------- BER ----------
figure; hold on; grid on; legend show;
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');
for mi = 1:numel(modes)
    m   = modes{mi};
    ber = nan(size(snrs));
    for k = 1:numel(snrs)
        key = matlab.lang.makeValidName(num2str(snrs(k)));  % 处理 -30 之类键名
        switch m
            case 'qpsk',   ber(k) = res.(m).(key).BER_QPSK   /(10000*4);  lbl = 'QPSK-bits BER (4bits)';
            case 'spatial',ber(k) = res.(m).(key).BER_AntSel /(10000*3);  lbl = 'Antenna-sel BER (3bits)';
            otherwise,     ber(k) = res.(m).(key).BER        /(10000*8);  lbl = 'All-bits BER (8bits)';
        end
    end
    semilogy(snrs, replaceZerosWithNaN(ber), markers.(m), 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', lbl);
end

% Swap-bit BER
ber_perm = nan(size(snrs));
for k = 1:numel(snrs)
    key = matlab.lang.makeValidName(num2str(snrs(k)));
    ber_perm(k) = res.spatial.(key).BER_perm /(10000*1);
end
semilogy(snrs, replaceZerosWithNaN(ber_perm), marker_perm, 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Swap-bit BER (1bit)');

xlabel('SNR [dB]'); ylabel('BER');

% ---------- SER ----------
figure; hold on; grid on; legend show;
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');
for mi = 1:numel(modes)
    m   = modes{mi};
    ser = nan(size(snrs));
    for k = 1:numel(snrs)
        key = matlab.lang.makeValidName(num2str(snrs(k)));
        switch m
            case 'qpsk',   ser(k) = res.(m).(key).SER_QPSK   /10000; lbl = 'QPSK-symbol SER';
            case 'spatial',ser(k) = res.(m).(key).SER_AntSel /10000; lbl = 'Antenna-sel SER';
            otherwise,     ser(k) = res.(m).(key).SER        /10000; lbl = 'All-symbol SER';
        end
    end
    semilogy(snrs, replaceZerosWithNaN(ser), markers.(m), 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', lbl);
end

% Swap-bit SER
ser_perm = nan(size(snrs));
for k = 1:numel(snrs)
    key = matlab.lang.makeValidName(num2str(snrs(k)));
    ser_perm(k) = res.spatial.(key).SER_perm /10000;
end
semilogy(snrs, replaceZerosWithNaN(ser_perm), marker_perm, 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Swap-bit SER');

xlabel('SNR [dB]'); ylabel('SER');

%% --------- 辅助函数 ---------
function v = replaceZerosWithNaN(v)
% 把 0 置 NaN，便于对数坐标不报错
v(v==0) = NaN;
end
