%% Plot_Combined.m —— 合并绘制 Mine.json 与 FRaC.json 的 BER / SER 曲线
%
%  说明:
%  - 该脚本将 Plot_Mine.m 和 Plot_FRaC.m 的功能合并.
%  - 它会在两个独立的图窗中分别绘制 BER 和 SER 曲线, 每个图窗包含来自两个 JSON 文件的数据.
%  - 需要 MATLAB R2016b+ (因其支持 jsondecode 函数).
%--------------------------------------------------------------------------

clear; clc; close all

%% ---------- ① 数据加载 ----------
% 加载两个 JSON 文件的数据
res_mine = jsondecode( fileread("Mine.json") );
res_frac = jsondecode( fileread("FRaC.json") );

% 定义信噪比轴 (两个文件通用)
snrs = -30:5:-5;

% 设置全局默认图形背景为白色
set(0,'defaultfigurecolor','w');


%% ---------- ② BER 曲线绘制 (合并) ----------
figure; hold on; grid on;
legend show; % 使用 'show' 以便自动选择最佳位置
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');

% ----- Plot_Mine.m 的 BER 数据 -----
ber_all_mine  = res_mine.case1_both.BER;
ber_tone_mine = res_mine.case2_tone_only.BER;
ber_qpsk_mine = res_mine.case3_qpsk_only.BER;

semilogy(snrs, ber_all_mine,  'o-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Prop.: All');
semilogy(snrs, ber_tone_mine, 'x-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Prop.: Delay-only');
semilogy(snrs, ber_qpsk_mine, 's-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Prop.: QPSK-only');

% ----- Plot_FRaC.m 的 BER 数据 -----
modes   = {'spatial','qpsk','all'};
markers = struct('spatial','s--','qpsk','^--','all','o--'); % 更改线型以区分
marker_perm = 'd--';

for mi = 1:numel(modes)
    m   = modes{mi};
    ber = nan(size(snrs));
    for k = 1:numel(snrs)
        key = matlab.lang.makeValidName(num2str(snrs(k)));
        switch m
            case 'qpsk',   ber(k) = res_frac.(m).(key).BER_QPSK   /(10000*4);  lbl = 'FRaC: QPSK-bits';
            case 'spatial',ber(k) = res_frac.(m).(key).BER_AntSel /(10000*3);  lbl = 'FRaC: Antenna-sel';
            otherwise,     ber(k) = res_frac.(m).(key).BER        /(10000*8);  lbl = 'FRaC: All-bits';
        end
    end
    semilogy(snrs, replaceZerosWithNaN(ber), markers.(m), 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', lbl);
end

% FRaC Swap-bit BER
ber_perm = nan(size(snrs));
for k = 1:numel(snrs)
    key = matlab.lang.makeValidName(num2str(snrs(k)));
    ber_perm(k) = res_frac.spatial.(key).BER_perm /(10000*1);
end
semilogy(snrs, replaceZerosWithNaN(ber_perm), marker_perm, 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','FRaC: Swap-bit');

xlabel('SNR [dB]'); ylabel('BER');
hold off;

%% ---------- ③ SER 曲线绘制 (合并) ----------
figure; hold on; grid on;
legend show;
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');

% ----- Plot_Mine.m 的 SER 数据 -----
ser_all_mine  = res_mine.case1_both.SER;
ser_tone_mine = res_mine.case2_tone_only.SER_tone;
ser_qpsk_mine = res_mine.case3_qpsk_only.SER_qpsk;

semilogy(snrs, ser_all_mine,  'o-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Prop.: All');
semilogy(snrs, ser_tone_mine, 'x-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Prop.: Delay-only');
semilogy(snrs, ser_qpsk_mine, 's-', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','Prop.: QPSK-only');

% ----- Plot_FRaC.m 的 SER 数据 -----
% (使用与 BER 部分相同的 modes 和 markers 变量)
for mi = 1:numel(modes)
    m   = modes{mi};
    ser = nan(size(snrs));
    for k = 1:numel(snrs)
        key = matlab.lang.makeValidName(num2str(snrs(k)));
        switch m
            case 'qpsk',   ser(k) = res_frac.(m).(key).SER_QPSK   /10000; lbl = 'FRaC: QPSK-only';
            case 'spatial',ser(k) = res_frac.(m).(key).SER_AntSel /10000; lbl = 'FRaC: Antenna-sel';
            otherwise,     ser(k) = res_frac.(m).(key).SER        /10000; lbl = 'FRaC: All';
        end
    end
    semilogy(snrs, replaceZerosWithNaN(ser), markers.(m), 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', lbl);
end

% FRaC Swap-bit SER
ser_perm = nan(size(snrs));
for k = 1:numel(snrs)
    key = matlab.lang.makeValidName(num2str(snrs(k)));
    ser_perm(k) = res_frac.spatial.(key).SER_perm /10000;
end
semilogy(snrs, replaceZerosWithNaN(ser_perm), marker_perm, 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName','FRaC: Swap-bit');

xlabel('SNR [dB]'); ylabel('SER');
hold off;

%% ---------- 辅助函数 ----------
function v = replaceZerosWithNaN(v)
% 把向量中的 0 替换为 NaN, 便于在对数坐标中绘图时不显示
v(v==0) = NaN;
end