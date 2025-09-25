clc; clear;

s = what; %look in current directory
%s=what('dir') %change dir for your directory name 
matfiles=s.mat;
nSeeds = numel(matfiles);

iter_num = 21;
avgEfficiencyDQ = zeros(iter_num,1);
avgEfficiencyQQ = zeros(iter_num,1);

for a=1:nSeeds
    load(char(matfiles(a)));    
    avgEfficiencyDQ = avgEfficiencyDQ + efficiencyDQ;
    avgEfficiencyQQ = avgEfficiencyQQ + efficiencyQQ;
end

avgEfficiencyDQ = avgEfficiencyDQ/nSeeds;
avgEfficiencyQQ = avgEfficiencyQQ/nSeeds;

figure; hold on

% plot(200,0,'-b*')
% plot(300,0,'-r>')

% plot([0 5:5:20],avgEfficiencyDQ([1 6:5:21]),'b--')
% plot([0 5:5:20],avgEfficiencyQQ([1 6:5:21]),'r--')

plot(0:20,avgEfficiencyDQ)
plot(0:20,avgEfficiencyQQ,'r')