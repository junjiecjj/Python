clc; clear;

s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

iter_num = 11;
avgConvergeWMMSE = zeros(iter_num,1);
avgConvergeNewton = zeros(iter_num,1);
avgConvergeFP = zeros(iter_num,1);
avgConvergeSCALE = zeros(iter_num,1);

for a=1:nSeeds
    load(char(matfiles(a)));    
    avgConvergeWMMSE = avgConvergeWMMSE + convergeWMMSE;
    avgConvergeNewton = avgConvergeNewton + convergeNewton;
    avgConvergeFP = avgConvergeFP + convergeFP;
    avgConvergeSCALE = avgConvergeSCALE + convergeSCALE;
end

avgConvergeWMMSE = avgConvergeWMMSE/nSeeds;
avgConvergeNewton = avgConvergeNewton/nSeeds;
avgConvergeFP = avgConvergeFP/nSeeds;
avgConvergeSCALE = avgConvergeSCALE/nSeeds;

figure; hold on

% plot(100,0,'-go')
% plot(200,0,'-r>')
% plot(300,0,'-b*')
% 
% plot([0 5:5:30],avgConvergeWMMSE([1 6:5:31]),'go')
% plot([0 5:5:30],avgConvergeNewton([1 6:5:31]),'r>')
% plot([0 5:5:30],avgConvergeFP([1 6:5:31]),'b*')

plot(0:10,avgConvergeWMMSE,'b')
plot(0:10,avgConvergeNewton,'r')
plot(0:10,avgConvergeFP,'g')
plot(0:10,avgConvergeSCALE,'c')