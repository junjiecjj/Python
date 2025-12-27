clear
clc
Factor = 1e6;
bandwidth = 10;
numBS = 7;
numUser = 2*3*7; % Q*L num sector * numUserin each sector
noise = 10^((-169-30)/10)*Factor;
numSlot = 1;
maxPower = ones(1,numBS)*10^((-47-30)/10)*Factor;

numTone = 1;
L = numBS; Q = 6; w = ones(Q,L); 
max_iter = 2*1e1; % max iteration of CQT
sigma = 1e-8; M = 128; N = 4; Pmax = 1e2;
test_num = 1; % number of random tests
iter_times = 5; % the max iteration of NQT and EQT is iter_times*max_iter
mimoPattern = [M,N]; % [tx,rx]
All_iter_results1 = zeros(max_iter+1,test_num);
All_time1 = zeros(max_iter,test_num); 

All_iter_results2 = zeros(iter_times*max_iter+1,test_num);
All_time2 = zeros(iter_times*max_iter,test_num);

All_iter_results3 = zeros(iter_times*max_iter+1,test_num);
All_time3 = zeros(iter_times*max_iter,test_num);

for test = 1:test_num
    [chn, distPathLoss ] = GenerateNetwork7(bandwidth, numBS, numUser,mimoPattern,numTone);
    chn = squeeze(chn);
    V = Generate_V(M,Q,L,Pmax); % Randomly initialize V
    [All_iter_results1(:,test),  All_time1(:,test)] = CQT(max_iter,sigma,M,N,L,Q,chn,w,Pmax,V);
    [All_iter_results2(:,test),  All_time2(:,test)]= NQT(iter_times*max_iter,sigma,M,N,L,Q,chn,w,Pmax,V);
    [All_iter_results3(:,test),  All_time3(:,test)] = EQT(iter_times*max_iter,sigma,M,N,L,Q,chn,w,Pmax,V);
end

ave_iter1 = sum(All_iter_results1(:,1:test_num),2)/test_num;
ave_time1 = sum(All_time1(:,1:test_num),2)/test_num;

ave_iter2 = sum(All_iter_results2(:,1:test_num),2)/test_num;
ave_time2 = sum(All_time2(:,1:test_num),2)/test_num;

ave_iter3 = sum(All_iter_results3(:,1:test_num),2)/test_num;
ave_time3 = sum(All_time3(:,1:test_num),2)/test_num;



