addpath('..\Data\')
clear;clc
%% B_640_T_512
figure
set(0,'defaultfigurecolor','w');
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');
box on; grid on;hold on; xlabel('SNR/dB');ylabel('BER')
BER_Combine(1:11) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;
load('..\Data\B_640_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
fields = fieldnames(BER);
for i = 1:length(BER)
    for j = 1:length(fields)
        field = fields{j};
        BER_Combine(i).(field) = BER_Combine(i).(field) + BER(i).(field);
    end
end
TotRng = TotRng + Nit*NumIMRngBit;
TotVel = TotVel + Nit*NumIMVelBit;
TotQAM = TotQAM + Nit*NumQAMBit;

SNRdB = -35:-30;
ErrorRng = flip([BER_Combine.NumIMRngError]);
ErrorRng = ErrorRng(1:6);
ErrorVel = flip([BER_Combine.NumIMVelError]);
ErrorVel = ErrorVel(1:6);
ErrorQAM = flip([BER_Combine.NumQAMError]);
ErrorQAM = ErrorQAM(1:6);
p1 = plot(SNRdB, ErrorRng / TotRng, ...
    'b->', 'LineWidth', 2, 'MarkerSize', 8);
p2 = plot(SNRdB, ErrorVel / TotVel, ...
    'b-<', 'LineWidth', 2, 'MarkerSize', 8);
p3 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'b-o', 'LineWidth', 2, 'MarkerSize', 8);
%% B_320_T_512
BER_Combine(1:11) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;
load('..\Data\B_320_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
fields = fieldnames(BER);
for i = 1:length(BER)
    for j = 1:length(fields)
        field = fields{j};
        BER_Combine(i).(field) = BER_Combine(i).(field) + BER(i).(field);
    end
end
TotRng = TotRng + Nit*NumIMRngBit;
TotVel = TotVel + Nit*NumIMVelBit;
TotQAM = TotQAM + Nit*NumQAMBit;

SNRdB = -35:-27;
ErrorRng = flip([BER_Combine.NumIMRngError]);
ErrorRng = ErrorRng(1:9);
ErrorRng(1:5) = TotRng;
ErrorVel = flip([BER_Combine.NumIMVelError]);
ErrorVel = ErrorVel(1:9);
ErrorVel(1:5) = TotVel;
ErrorQAM = flip([BER_Combine.NumQAMError]);
ErrorQAM = ErrorQAM(1:9);
ErrorQAM(1:5) = TotQAM;
p4 = plot(SNRdB, ErrorRng / TotRng, ...
    'r->', 'LineWidth', 2, 'MarkerSize', 8);
p5 = plot(SNRdB, ErrorVel / TotVel, ...
    'r-<', 'LineWidth', 2, 'MarkerSize', 8);
p6 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'r-o', 'LineWidth', 2, 'MarkerSize', 8);
%% B_160_T_512
BER_Combine(1:11) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;
load('..\Data\B_160_T_512_SNR_25_35_Nit_1e4_rng_666.mat')
fields = fieldnames(BER);
for i = 1:length(BER)
    for j = 1:length(fields)
        field = fields{j};
        BER_Combine(i).(field) = BER_Combine(i).(field) + BER(i).(field);
    end
end
TotRng = TotRng + Nit*NumIMRngBit;
TotVel = TotVel + Nit*NumIMVelBit;
TotQAM = TotQAM + Nit*NumQAMBit;

SNRdB = -35:-25;
ErrorRng = flip([BER_Combine.NumIMRngError]);
ErrorVel = flip([BER_Combine.NumIMVelError]);
ErrorQAM = flip([BER_Combine.NumQAMError]);
p7 = plot(SNRdB, ErrorRng / TotRng, ...
    'g->', 'LineWidth', 2, 'MarkerSize', 8);
p8 = plot(SNRdB, ErrorVel / TotVel, ...
    'g-<', 'LineWidth', 2, 'MarkerSize', 8);
p9 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'g-o', 'LineWidth', 2, 'MarkerSize', 8);
%% B_640_T_256
BER_Combine(1:11) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;
load('..\Data\B_640_T_256_SNR_25_35_Nit_1e4_rng_666.mat')
fields = fieldnames(BER);
for i = 1:length(BER)
    for j = 1:length(fields)
        field = fields{j};
        BER_Combine(i).(field) = BER_Combine(i).(field) + BER(i).(field);
    end
end
TotRng = TotRng + Nit*NumIMRngBit;
TotVel = TotVel + Nit*NumIMVelBit;
TotQAM = TotQAM + Nit*NumQAMBit;

SNRdB = -35:-27;
ErrorRng = flip([BER_Combine.NumIMRngError]);
ErrorRng = ErrorRng(1:9);
ErrorRng(1:5) = TotRng;
ErrorVel = flip([BER_Combine.NumIMVelError]);
ErrorVel = ErrorVel(1:9);
ErrorVel(1:5) = TotVel;
ErrorQAM = flip([BER_Combine.NumQAMError]);
ErrorQAM = ErrorQAM(1:9);
ErrorQAM(1:5) = TotQAM;
p10 = plot(SNRdB, ErrorRng / TotRng, ...
    'c-->', 'LineWidth', 2, 'MarkerSize', 8);
p11 = plot(SNRdB, ErrorVel / TotVel, ...
    'c--<', 'LineWidth', 2, 'MarkerSize', 8);
p12 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'c--o', 'LineWidth', 2, 'MarkerSize', 8);
%% legend
ah1=axes('position',get(gca,'position'),'visible','off'); 
l1 = legend(ah1, [p1, p2, p3],...
    {['Delay:'      repmat(char(8202),1,22)   'B=640 M, T_c=51.2 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)   'B=640 M, T_c=51.2 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)   'B=640 M, T_c=51.2 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l1.FontSize = 12;   
l1.FontName = 'Times New Roman';

ah2=axes('position',get(gca,'position'),'visible','off'); 
l2 = legend(ah2, [p4, p5, p6],...
    {['Delay:'      repmat(char(8202),1,22)   'B=320 M, T_c=51.2 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)   'B=320 M, T_c=51.2 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)   'B=320 M, T_c=51.2 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l2.FontSize = 12;   
l2.FontName = 'Times New Roman';

ah3=axes('position',get(gca,'position'),'visible','off'); 
l3 = legend(ah3, [p7, p8, p9],...
    {['Delay:'      repmat(char(8202),1,22)  'B=160 M, T_c=51.2 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)  'B=160 M, T_c=51.2 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)   'B=160 M, T_c=51.2 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l3.FontSize = 12;   
l3.FontName = 'Times New Roman';

ah4=axes('position',get(gca,'position'),'visible','off'); 
l4 = legend(ah4, [p10, p11, p12],...
    {['Delay:'      repmat(char(8202),1,22)  'B=640 M, T_c=25.6 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)  'B=640 M, T_c=25.6 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)   'B=640 M, T_c=25.6 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l4.FontSize = 12;   
l4.FontName = 'Times New Roman';