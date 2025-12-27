cvx_solver SDPT3

clc; clear;
% throw random seeds (for older version matlab)
ctime = datestr(now,30);
tseed = str2double(ctime((end-5):end));
rand('seed',tseed)
randn('seed',tseed)

rand()
randn()

Factor = 1e6;
bandwidth = 10; % MHz
numBS = 7; % num BS
numUser = 1*3*7; % num user
% numBS = 50;
% numUser = numBS;
noise = 10^((-169-30)/10)*Factor;
numSlot = 1;%200;
maxPower = ones(1,numBS)*10^((-47-30)/10)*Factor;
mimoPattern = [ 1,1 ];%[ 4,2 ];%[ 4,2 ]; % [ tx, rx ]
numTone = 4;

% [ chn, distPathLoss ] = generateNetwork( numBS, mimoPattern, numTone );
[ chn, distPathLoss ] = GenerateNetwork7(bandwidth, numBS, numUser, mimoPattern, numTone );
association = decideAssociation(numBS, numUser, distPathLoss);

numAlgorithm = 4;
algorithm = cell(numAlgorithm,1);
algorithm{1} = 'FP';
algorithm{2} = 'WMMSE'; % wmmse
algorithm{3} = 'Newton';
algorithm{4} = 'SCALE';

% initial V
L = numBS;
M = mimoPattern(1); 
T = numTone;
initV = nan(M,T,L,M); % V(:,z,j,s) is tx at tone z, BS j, stream s
for j = 1:L
    for s = 1:M
        for z = 1:T
%             V(:,z,j,s) = (rand(M,1)+1i*rand(M,1))/sqrt(2)*sqrt(maxPower(j)/M/M/T);
            initV(:,z,j,s) = ones(M,1)*sqrt(maxPower(j)/M/M/T);
        end
    end
end 

global convergeWMMSE
global convergeFP
global convergeNewton   
global convergeSCALE

obj = cell(numAlgorithm);
for alg = [1 2 3 4]%1:numAlgorithm   
    obj{alg} = MyClass(bandwidth, numTone, numBS, numUser, noise, numSlot, mimoPattern, chn, distPathLoss, maxPower, association, algorithm{alg}, initV);
    obj{alg}.computeAvgRate();
end

figure; hold on
plot(convergeFP,'g')
plot(convergeWMMSE)
plot(convergeNewton,'r')
plot(convergeSCALE,'c')


% save(strrep(strrep(num2str(clock),' ',''),'.','_'), 'convergeWMMSE','convergeSCALE','convergeNewton','convergeFP');