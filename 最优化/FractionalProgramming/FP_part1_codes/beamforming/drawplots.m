clc; clear;
K = 4*3*7;
numMIMO = 1;
[avgRateProposed, avgRateWMMSE, avgRateFixed] = deal(zeros(K,numMIMO));
[numScheduleProposed, numScheduleWMMSE, numScheduleFixed] = deal(zeros(1,numMIMO));
s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

for a=1:nSeeds
    load(char(matfiles(a)));
    for p = 1:numMIMO      
        avgRateProposed(:,p) = avgRateProposed(:,p) + sort(avgRateArray{1});
        avgRateWMMSE(:,p) = avgRateWMMSE(:,p) + sort(avgRateArray{2});
        avgRateFixed(:,p) = avgRateFixed(:,p) + sort(avgRateArray{3});
        
        numScheduleProposed(p) = numScheduleProposed(p) + numScheduleArray{1};
        numScheduleWMMSE(p) = numScheduleWMMSE(p) + numScheduleArray{2};
        numScheduleFixed(p) = numScheduleFixed(p) + numScheduleArray{3};
    end
end

for p = 1:numMIMO
    avgRateProposed(:,p) = avgRateProposed(:,p)/nSeeds;
    avgRateWMMSE(:,p) = avgRateWMMSE(:,p)/nSeeds;
    avgRateFixed(:,p) = avgRateFixed(:,p)/nSeeds;
end

utilityProposed = sum(log(avgRateProposed));
utilityWMMSE = sum(log(avgRateWMMSE));
utilityFixed = sum(log(avgRateFixed));

utilityProposed
utilityWMMSE
utilityFixed

numScheduleProposed/nSeeds
numScheduleWMMSE/nSeeds
numScheduleFixed/nSeeds

figure; hold on
[x,y]=ecdf(avgRateProposed(:,1)); plot(y,x,'b')
[x,y]=ecdf(avgRateWMMSE(:,1)); plot(y,x,'r')
[x,y]=ecdf(avgRateFixed(:,1)); plot(y,x,'g')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
open('rcdf.fig')
h = get(gca,'Children');
xdata = get(h,'Xdata');
ydata = get(h,'Ydata');

figure; hold on
grid on
plot(100,1,'b-*')
plot(100,1,'r-o')
plot(100,1,'g->')
% plot(100,1,'k-s')
% plot(100,1,'b-x')
% plot(100,1,'r-<')
% plot(100,1,'k-o')
% plot(100,1,'b-+')
% plot(100,1,'r->')

legend('Proposed', 'WMMSE', 'Fixed interference')

% legend('Fixed (2X1)','Proposed (2X1)','MMSE (2X1)',...
%     'Fixed (2X2)','Proposed (2X2)','MMSE (2X2)',...
%     'Fixed (4X2)','Proposed (4X2)','MMSE (4X2)');

%!!

plot(cell2mat(xdata(1)),cell2mat(ydata(1)),'-g')
plot(cell2mat(xdata(2)),cell2mat(ydata(2)),'-r')
plot(cell2mat(xdata(3)),cell2mat(ydata(3)),'-b')
% plot(cell2mat(xdata(4)),cell2mat(ydata(4)),'-r')
% plot(cell2mat(xdata(5)),cell2mat(ydata(5)),'-b')
% plot(cell2mat(xdata(6)),cell2mat(ydata(6)),'-k')
% plot(cell2mat(xdata(7)),cell2mat(ydata(4)),'-r')
% plot(cell2mat(xdata(8)),cell2mat(ydata(5)),'-b')
% plot(cell2mat(xdata(9)),cell2mat(ydata(6)),'-k')

len = length(cell2mat(xdata(1)));
x = cell2mat(xdata(1)); y = cell2mat(ydata(1)); plot(x(1:5:len),y(1:5:len),'g>')
x = cell2mat(xdata(2)); y = cell2mat(ydata(2)); plot(x(1:5:len),y(1:5:len),'ro')
x = cell2mat(xdata(3)); y = cell2mat(ydata(3)); plot(x(1:5:len),y(1:5:len),'b*')
% x = cell2mat(xdata(4)); y = cell2mat(ydata(4)); plot(x(1:5:len),y(1:5:len),'r<')
% x = cell2mat(xdata(5)); y = cell2mat(ydata(5)); plot(x(1:5:len),y(1:5:len),'bx')
% x = cell2mat(xdata(6)); y = cell2mat(ydata(6)); plot(x(1:5:len),y(1:5:len),'ks')
% x = cell2mat(xdata(7)); y = cell2mat(ydata(7)); plot(x(1:5:len),y(1:5:len),'r^')
% x = cell2mat(xdata(8)); y = cell2mat(ydata(8)); plot(x(1:5:len),y(1:5:len),'b*')
% x = cell2mat(xdata(9)); y = cell2mat(ydata(9)); plot(x(1:5:len),y(1:5:len),'kd')

xlabel('Data rate (Mbps)'); ylabel('Cumulative distribution')