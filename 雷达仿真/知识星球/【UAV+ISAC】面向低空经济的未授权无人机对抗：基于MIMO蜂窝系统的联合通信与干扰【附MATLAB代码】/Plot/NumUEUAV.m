% plot
close all

load("..\Data\NumUEUAV_it_2000.mat")
ErrorLength = length(ErrorIt_Index);
for i_error = 1:ErrorLength
    P_12(ErrorCase_Index(i_error), ErrorIt_Index(i_error)) = NaN;
    P_123(ErrorCase_Index(i_error), ErrorIt_Index(i_error)) = NaN;
    MaxErrorRate(ErrorCase_Index(i_error), ErrorIt_Index(i_error)) = NaN;
    MaxErrorSINR(ErrorCase_Index(i_error), ErrorIt_Index(i_error)) = NaN;
end
%% |    Absolute Power Error
set(0,'defaultfigurecolor','w') 
figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(10*log10(abs(P_12(1,:)-P_123(1,:))));
    set(p1,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p2  = cdfplot(10*log10(abs(P_12(2,:)-P_123(2,:))));
    set(p2,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p3  = cdfplot(10*log10(abs(P_12(3,:)-P_123(3,:))));
    set(p3,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p4  = cdfplot(10*log10(abs(P_12(4,:)-P_123(4,:))));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)   
p5  = cdfplot(10*log10(abs(P_12(5,:)-P_123(5,:))));
    set(p5,'LineStyle','-', 'Color', 'm', 'LineWidth', 3)   
xlabel('Power Error [dBm]');
ylabel('CDF');
title('') 
ah1=axes('position',get(gca,'position'),'visible','off'); 
l1 = legend([p1, p2, p3, p4, p5],{  
        'UE=2, UAV=2',...
        'UE=3, UAV=2',...
        'UE=3, UAV=8',...
        'UE=3, UAV=14',...
        'UE=4, UAV=2'},...
        'Interpreter','tex',...
        'NumColumns', 1);
l1.FontSize = 16;
l1.FontName = 'Times New Roman';
%% |    Normlized Power Error
set(0,'defaultfigurecolor','w') 
figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(10*log10(abs(P_12(1,:)-P_123(1,:))./P_123(1,:)));
    set(p1,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p2  = cdfplot(10*log10(abs(P_12(2,:)-P_123(2,:))./P_123(2,:)));
    set(p2,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p3  = cdfplot(10*log10(abs(P_12(3,:)-P_123(3,:))./P_123(3,:)));
    set(p3,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p4  = cdfplot(10*log10(abs(P_12(4,:)-P_123(4,:))./P_123(4,:)));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)   
p5  = cdfplot(10*log10(abs(P_12(5,:)-P_123(5,:))./P_123(5,:)));
    set(p5,'LineStyle','-', 'Color', 'm', 'LineWidth', 3)   
xlabel('Normlized Power Error [dB]');
ylabel('CDF');
title('') 
ah2=axes('position',get(gca,'position'),'visible','off'); 
l2 = legend([p1, p2, p3, p4, p5],{        
        'UE=2, UAV=2',...
        'UE=3, UAV=2',...
        'UE=3, UAV=8',...
        'UE=3, UAV=14',...
        'UE=4, UAV=2'},...
        'Interpreter','tex',...
        'NumColumns', 1);
l2.FontSize = 16;
l2.FontName = 'Times New Roman';
%% |    Max Rate Error
set(0,'defaultfigurecolor','w') 
figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(MaxErrorRate(1,:));
    set(p1,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p2  = cdfplot(MaxErrorRate(2,:));
    set(p2,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p3  = cdfplot(MaxErrorRate(3,:));
    set(p3,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p4  = cdfplot(MaxErrorRate(4,:));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)   
p5  = cdfplot(MaxErrorRate(5,:));
    set(p5,'LineStyle','-', 'Color', 'm', 'LineWidth', 3)   
xlim([0 2])
xlabel('Rate Error [bits/(s{\cdot}Hz)]');
ylabel('CDF');
title('') 
ah3=axes('position',get(gca,'position'),'visible','off'); 
l3 = legend([p1, p2, p3, p4, p5],{  
        'UE=2, UAV=2',...
        'UE=3, UAV=2',...
        'UE=3, UAV=8',...
        'UE=3, UAV=14',...
        'UE=4, UAV=2'},...
        'Interpreter','tex',...
        'NumColumns', 1);
l3.FontSize = 16;
l3.FontName = 'Times New Roman';
%% |    Max SINR Error
set(0,'defaultfigurecolor','w') 
figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(MaxErrorSINR(1,:));
    set(p1,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p2  = cdfplot(MaxErrorSINR(2,:));
    set(p2,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p3  = cdfplot(MaxErrorSINR(3,:));
    set(p3,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p4  = cdfplot(MaxErrorSINR(4,:));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)   
p5  = cdfplot(MaxErrorSINR(5,:));
    set(p5,'LineStyle','-', 'Color', 'm', 'LineWidth', 3)   
xlabel('SINR Error [dB]');
ylabel('CDF');
title('') 
ah4=axes('position',get(gca,'position'),'visible','off'); 
l4 = legend([p1, p2, p3, p4, p5],{  
        'UE=2, UAV=2',...
        'UE=3, UAV=2',...
        'UE=3, UAV=8',...
        'UE=3, UAV=14',...
        'UE=4, UAV=2'},...
        'Interpreter','tex',...
        'NumColumns', 1);
l4.FontSize = 16;
l4.FontName = 'Times New Roman';

%% |    Pt vs Nt CDF
load("..\Data\CI_NumUEUAV_it_5000.mat")
set(0,'defaultfigurecolor','w') 
figure; hold on; box on; grid on;
set(gca,'FontName','Times New Roman','FontSize',16);
p1  = cdfplot(10*log10(P_123(1,:)));
    set(p1,'LineStyle','-', 'Color', 'r', 'LineWidth', 3) 
p2  = cdfplot(10*log10(P_123(2,:)));
    set(p2,'LineStyle','-', 'Color', 'g', 'LineWidth', 3) 
p3  = cdfplot(10*log10(P_123(3,:)));
    set(p3,'LineStyle','-', 'Color', 'b', 'LineWidth', 3) 
p4  = cdfplot(10*log10(P_123(4,:)));
    set(p4,'LineStyle','-', 'Color', 'c', 'LineWidth', 3)   
p5  = cdfplot(10*log10(P_123(5,:)));
    set(p5,'LineStyle','-', 'Color', 'm', 'LineWidth', 3)  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p11  = cdfplot(10*log10(P(1,:)));
    set(p11,'LineStyle','--', 'Color', 'r', 'LineWidth', 2) 
p22  = cdfplot(10*log10(P(2,:)));
    set(p22,'LineStyle','--', 'Color', 'g', 'LineWidth', 2) 
p33  = cdfplot(10*log10(P(3,:)));
    set(p33,'LineStyle','--', 'Color', 'b', 'LineWidth', 2) 
% p44  = cdfplot(10*log10(P(4,:)));
%     set(p44,'LineStyle','--', 'Color', 'c', 'LineWidth', 2)   
p55  = cdfplot(10*log10(P(5,:)));
    set(p55,'LineStyle','--', 'Color', 'm', 'LineWidth', 2) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
xlabel('Transmit Power [dBm]');
ylabel('CDF');
% xlim([-13 60])
title('') 
ah1=axes('position',get(gca,'position'),'visible','off'); 
l1 = legend([p1, p2, p3, p4, p5, ...
            p11, p22, p33, p55],{  
        'UE=2, UAV=2',...
        'UE=3, UAV=2',...
        'UE=3, UAV=8',...
        'UE=3, UAV=14',...
        'UE=4, UAV=2',...
        'UE=2, UAV=2, CI',...
        'UE=3, UAV=2, CI',...
        'UE=3, UAV=8, CI',...
        'UE=4, UAV=2, CI'},...
        'Interpreter','tex', 'Box','on',...
        'NumColumns', 2);
l1.FontSize = 16;
l1.FontName = 'Times New Roman';