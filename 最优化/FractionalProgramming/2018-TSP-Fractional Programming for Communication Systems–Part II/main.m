clc; clear;
% set random seeds (for older version matlab)
ctime = datestr(now,30);
tseed = str2double(ctime((end-5):end));
rand('seed',tseed)
randn('seed',tseed)
% rng(tseed)

rand()
randn()

bandwidth = 10; % MHz
numBS = 7; % num BS
numUser = 4*3*7; % num user
noise = 10^((-171-30)/10);
numSlot = 500;
maxPower = ones(numUser,1)*10^((-47-30)/10);
mimoPattern = [ 1,1 ];
% mimoPattern = [ 1,2; 2,2; 2,4 ];
numPattern = size(mimoPattern,1);

[chnArray, chnMagnitude] = GenerateNetwork7( numBS, numUser, mimoPattern );
association = decideAssociation(numBS, numUser, chnMagnitude);

numAlgorithm = 3;
algorithm = cell(numAlgorithm,1);
algorithm{1} = 'proposed';
algorithm{2} = 'WMMSE';
algorithm{3} = 'fixed';

for iter = 1:1
    obj = cell(numAlgorithm,numPattern);
    avgRateArray = cell(numAlgorithm,numPattern);
    numScheduleArray = cell(numAlgorithm, numPattern);
    for p = 1:numPattern
        pattern = mimoPattern(p,:);
        numTxAnte = pattern(1); numRxAnte = pattern(2);    
        chn = chnArray{p};
        for a = 1:numAlgorithm   
            obj{a,p} = MyClass(bandwidth, numBS, numUser, noise, numSlot, numTxAnte,...
                numRxAnte, chn, chnMagnitude, maxPower, association, algorithm{a});
            [ numScheduleArray{a,p}, avgRateArray{a,p} ]= obj{a,p}.computeAvgRate();
        end
    end

    save(strrep(strrep(num2str(clock),' ',''),'.','_'), 'avgRateArray','numScheduleArray');
end

figure; hold on
[x, y] = ecdf(avgRateArray{1,1}); plot(y,x,'b')
[x, y] = ecdf(avgRateArray{2,1}); plot(y,x,'r')
[x, y] = ecdf(avgRateArray{3,1}); plot(y,x,'g')
% 
% utiltiy1 = sum(log(avgRateArray{1,1}))
% utiltiy1 = sum(log(avgRateArray{2,1}))
% utiltiy1 = sum(log(avgRateArray{3,1}))
