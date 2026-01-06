addpath('..\Data\')
clear;clc
%% B_640_T_512
figure
set(0,'defaultfigurecolor','w');
set(gca,'FontName','Times New Roman','FontSize',16);
set(gca, 'YScale', 'log');
ylim([1e-4 1])
box on; grid on;hold on; xlabel('SNR [dB]');ylabel(' SER')
SER_Combine(1:11) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;
load('..\Data\BER_Hitrate\B_640_T_512.mat')
fields = fieldnames( SER);
for i = 1:length( SER)
    for j = 1:length(fields)
        field = fields{j};
        SER_Combine(i).(field) = SER_Combine(i).(field) +  SER(i).(field);
    end
end
TotRng = TotRng + Nit ;
TotVel = TotVel + Nit ;
TotQAM = TotQAM + Nit ;

SNRdB = -35:-30;
ErrorRng = flip([SER_Combine.NumIMRngError]);
ErrorRng = ErrorRng(1:6);
ErrorVel = flip([SER_Combine.NumIMVelError]);
ErrorVel = ErrorVel(1:6);
ErrorQAM = flip([SER_Combine.NumQAMError]);
ErrorQAM = ErrorQAM(1:6);
p1 = plot(SNRdB, ErrorRng / TotRng, ...
    'b->', 'LineWidth', 2, 'MarkerSize', 8);
p2 = plot(SNRdB, ErrorVel / TotVel, ...
    'b-<', 'LineWidth', 2, 'MarkerSize', 8);
p3 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'b-o', 'LineWidth', 2, 'MarkerSize', 8);
%% B_640_T_256
SER_Combine(1:11) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;
load('..\Data\BER_Hitrate\B_640_T_256.mat')
fields = fieldnames( SER);
for i = 1:length( SER)
    for j = 1:length(fields)
        field = fields{j};
        SER_Combine(i).(field) = SER_Combine(i).(field) +  SER(i).(field);
    end
end
TotRng = TotRng + Nit ;
TotVel = TotVel + Nit ;
TotQAM = TotQAM + Nit ;

SNRdB = -35:-27;
ErrorRng = flip([SER_Combine.NumIMRngError]);
ErrorRng = ErrorRng(1:9);
ErrorVel = flip([SER_Combine.NumIMVelError]);
ErrorVel = ErrorVel(1:9);
ErrorQAM = flip([SER_Combine.NumQAMError]);
ErrorQAM = ErrorQAM(1:9);
p4 = plot(SNRdB, ErrorRng / TotRng, ...
    'r->', 'LineWidth', 2, 'MarkerSize', 8);
p5 = plot(SNRdB, ErrorVel / TotVel, ...
    'r-<', 'LineWidth', 2, 'MarkerSize', 8);
p6 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'r-o', 'LineWidth', 2, 'MarkerSize', 8);
%% B_320_T_512
SER_Combine(1:11) = struct(...
    'NumQAMError',0,...
    'NumIMRngError',0,...
    'NumIMVelError',0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;
load('..\Data\BER_Hitrate\B_320_T_512.mat')
fields = fieldnames( SER);
for i = 1:length( SER)
    for j = 1:length(fields)
        field = fields{j};
        SER_Combine(i).(field) = SER_Combine(i).(field) +  SER(i).(field);
    end
end
TotRng = TotRng + Nit ;
TotVel = TotVel + Nit ;
TotQAM = TotQAM + Nit ;

SNRdB = -35:-27;
ErrorRng = flip([SER_Combine.NumIMRngError]);
ErrorRng = ErrorRng(1:9);
ErrorVel = flip([SER_Combine.NumIMVelError]);
ErrorVel = ErrorVel(1:9);
ErrorQAM = flip([SER_Combine.NumQAMError]);
ErrorQAM = ErrorQAM(1:9);
p7 = plot(SNRdB, ErrorRng / TotRng, ...
    'g-->', 'LineWidth', 2, 'MarkerSize', 8);
p8 = plot(SNRdB, ErrorVel / TotVel, ...
    'g--<', 'LineWidth', 2, 'MarkerSize', 8);
p9 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'g--o', 'LineWidth', 2, 'MarkerSize', 8);
%% B_160_T_512
% 初始化结构体和计数器
SER_Combine(1:11) = struct(...
    'NumQAMError', 0, ...
    'NumIMRngError', 0, ...
    'NumIMVelError', 0);
TotRng = 0;
TotVel = 0;
TotQAM = 0;

% 加载 -35dB 到 -25dB 的数据
load('..\Data\BER_Hitrate\B_160_T_512.mat')
fields = fieldnames( SER);
for i = 1:length( SER)
    for j = 1:length(fields)
        field = fields{j};
        SER_Combine(i).(field) = SER_Combine(i).(field) +  SER(i).(field);
    end
end
TotRng = TotRng + Nit ;
TotVel = TotVel + Nit ;
TotQAM = TotQAM + Nit;

ErrorRng = flip([SER_Combine(1:11).NumIMRngError]);
ErrorVel = flip([SER_Combine(1:11).NumIMVelError]);
ErrorQAM = flip([SER_Combine(1:11).NumQAMError]);

SNRdB = -35:-25;
p10 = plot(SNRdB, ErrorRng / TotRng, ...
    'c-->', 'LineWidth', 2, 'MarkerSize', 8);
p11 = plot(SNRdB, ErrorVel / TotVel, ...
    'c--<', 'LineWidth', 2, 'MarkerSize', 8);
p12 = plot(SNRdB, ErrorQAM / TotQAM, ...
    'c--o', 'LineWidth', 2, 'MarkerSize', 8);
%% legend
ah1=axes('position',get(gca,'position'),'visible','off'); 
l1 = legend(ah1, [p1, p2, p3],...
    {['Delay:'      repmat(char(8202),1,22)   'B=640 MHz, T_c=51.2 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)    'B=640 MHz, T_c=51.2 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)    'B=640 MHz, T_c=51.2 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l1.FontSize = 12;   
l1.FontName = 'Times New Roman';

ah2=axes('position',get(gca,'position'),'visible','off'); 
l2 = legend(ah2, [p4, p5, p6],...
    {['Delay:'      repmat(char(8202),1,22)   'B=640 MHz, T_c=25.6 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)   'B=640 MHz, T_c=25.6 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)   'B=640 MHz, T_c=25.6 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l2.FontSize = 12;   
l2.FontName = 'Times New Roman';

ah3=axes('position',get(gca,'position'),'visible','off'); 
l3 = legend(ah3, [p7, p8, p9],...
    {['Delay:'      repmat(char(8202),1,22)  'B=320 MHz, T_c=51.2 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)   'B=320 MHz, T_c=51.2 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)   'B=320 MHz, T_c=51.2 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l3.FontSize = 12;   
l3.FontName = 'Times New Roman';

ah4=axes('position',get(gca,'position'),'visible','off'); 
l4 = legend(ah4, [p10, p11, p12],...
    {['Delay:'      repmat(char(8202),1,22)  'B=160 MHz, T_c=51.2 \mus'],...
     ['Doppler:'   repmat(char(8202),1,12)  'B=160 MHz, T_c=51.2 \mus'], ...
     ['Amplitude:'  repmat(char(8202),1,0)   'B=160 MHz, T_c=51.2 \mus']},...
    'Box','off', ...
    'Interpreter','tex');
l4.FontSize = 12;   
l4.FontName = 'Times New Roman';