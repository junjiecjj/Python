

clear all;
close all;
clc;


addpath(pwd);
cd manopt;
addpath(genpath(pwd));
cd ..;
addpath('./MO_AltMin');
addpath('./PE_AltMin');
Ns = 3;
NRF = 3;

SNR_dB = -35:5:5;
SNR = 10.^(SNR_dB./10);
Iterations = 20;
smax = length(SNR);% enable the parallel
count = 0;
for it = 1:Iterations
    count = count + 1;
    [H, Fopt, Wopt, At, Ar] = channel_realization(5, 10, Ns, 144, 36);

    %% proposed MO_AltMin algo.
    [ FRF, FBB ] = MO_AltMin(Fopt, NRF);
    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
    [ WRF, WBB ] = MO_AltMin(Wopt, NRF);
    for s = 1:smax
        R1(s,it) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));
        R_o(s,it) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(Wopt) * H * Fopt * Fopt' * H' * Wopt));
    end

    %% proposed PE_AltMin algo.
    [ FRF, FBB ] = PE_AltMin( Fopt, NRF);
    FBB = sqrt(Ns) * FBB / norm(FRF * FBB,'fro');
    [ WRF, WBB ] = PE_AltMin( Wopt, NRF);
    for s = 1:smax
        R2(s,it) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF * WBB) * H * FRF * FBB * FBB' * FRF' * H' * WRF * WBB));
    end
 
    count
end


width = 7;%设置图宽，这个不用改
height = 7*0.75;%设置图高，这个不用改
fontsize = 18;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
markersize = 10;%标记的大小，按照个人喜好设置。

h = figure(1);
fig(h, 'units','inches','width',width, 'height', height, 'font','Times New Roman','fontsize',fontsize);%这是用于裁剪figure的。需要把fig.m文件放在一个文件夹中

plot(SNR_dB, sum(real(R_o),2)/count,'k-o','LineWidth',1.5, 'markersize',10); hold on;
plot(SNR_dB, sum(real(R1),2)/count,'m-*','LineWidth',1.5, 'markersize',10); hold on;
plot(SNR_dB, sum(real(R2),2)/count,'b-s','LineWidth',1.5, 'markersize',10); hold on;
plot(SNR_dB, sum(real(R3),2)/count,'g-^','LineWidth',1.5, 'markersize',10); hold on;
plot(SNR_dB, sum(real(Rsdr),2)/count,'r-d','LineWidth',1.5, 'markersize',10); hold on;
plot(SNR_dB, sum(real(Rsic),2)/count,'c-v','LineWidth',1.5, 'markersize',10); hold on;
grid on;

%set(gca,'XMinorGrid','off'); % 关闭X轴的次网格
%set(gca,'XGrid','off','LineWidth',0.01); % 关闭X轴的网格
set(gca,'gridlinestyle','--','Gridalpha',0.2,'LineWidth',0.01,'Layer','bottom');

% gca表示对axes的设置；  gcf表示对figure的设置
%---------------------------------------------------------
scalesize = 28;
% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');

h_legend = legend('Optimal Digital Precoder', ...
                  'MO-AltMin', ...
                  'PE-AltMin', ...
                  'OMP algo',...
                  'SDR algo',...
                  'SIC algo' ...
                  );  %图例，与上面的曲线先后对应
legendsize = 22;
set(h_legend,'FontName','Times New Roman','FontSize',fontsize,'FontWeight','normal','Location','southwest');
set(h_legend,'Interpreter','latex');

labelsize = 28;
xlabel('SNR(dB)','FontName','Times New Roman','FontSize',fontsize,'FontWeight','normal','Color','k','Interpreter','latex');%横坐标标号,坐标轴label字体、字体大小
ylabel('Rate(bit/s/Hz)','FontName','Times New Roman','FontSize',fontsize,'FontWeight','normal','Color','k','Interpreter','latex');%纵坐标标号，坐标轴label字体、字体大小
%set(get(gca,'XLabel'),'FontSize',14);%图上文字为8 point或小5号
%set(get(gca,'YLabel'),'FontSize',14);
set(gca,'linewidth',1.5);       % 设置坐标轴粗细
set(gcf,'color','white');  % 设置背景是白色的 原先是灰色的 论文里面不好看