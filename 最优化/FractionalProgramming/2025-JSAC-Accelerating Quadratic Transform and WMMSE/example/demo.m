%% demo example
clc
clear


N = 5; %  
d = 9;
L = 4;
P = 10; 

iter_num = 2*1e3;
test_num = 1; 

All_iter_results1 = zeros(iter_num+1,test_num);
All_time1 = zeros(iter_num,test_num);

All_iter_results2 = zeros(iter_num+1,test_num);
All_time2 = zeros(iter_num,test_num);

All_iter_results3 = zeros(iter_num+1,test_num);
All_time3 = zeros(iter_num,test_num);

All_iter_results4 = zeros(iter_num+1,test_num);
All_time4 = zeros(iter_num,test_num);

All_iter_results5 = zeros(iter_num+1,test_num);
All_time5 = zeros(iter_num,test_num);

for t = 1:test_num
    t
    % randomly generate A and B
    A = (randn(d,d,N)+1i*randn(d,d,N))/sqrt(2);
    B = (randn(d,d,N,N)+1i*randn(d,d,N))/sqrt(2);
    % randomly initialize X
    X = (randn(d,L,N)+1i*randn(d,L,N))/sqrt(2);
    [All_iter_results1(:,t),All_time1(:,t)] = CQT( iter_num, P, N, d, L, A, B, X ); % Conventional quadratic transform 
    [All_iter_results2(:,t),All_time2(:,t)] = NQT( iter_num, P, N, d, L, A, B, X ); % Nonhomogeneou quadratic transform
    [All_iter_results3(:,t),All_time3(:,t)] = EQT( iter_num, P, N, d, L, A, B, X ); % Extrapolated quadratic transform
    [All_iter_results4(:,t),All_time4(:,t)] =  NQT_dismissing( iter_num, P, N, d, L, A, B, X ); % Nonhomogeneou QT with dismissing 
    [All_iter_results5(:,t),All_time5(:,t)] =  EQT_Polyak( iter_num, P, N, d, L, A, B, X ); % Polyak's extrapolation 
end
ave_iter1 = sum(All_iter_results1(:,1:test_num),2)/test_num;
ave_time1 = sum(All_time1(:,1:test_num),2)/test_num;

ave_iter2 = sum(All_iter_results2(:,1:test_num),2)/test_num;
ave_time2 = sum(All_time2(:,1:test_num),2)/test_num;

ave_iter3 = sum(All_iter_results3(:,1:test_num),2)/test_num;
ave_time3 = sum(All_time3(:,1:test_num),2)/test_num;

ave_iter4 = sum(All_iter_results4(:,1:test_num),2)/test_num;
ave_time4 = sum(All_time4(:,1:test_num),2)/test_num;

ave_iter5 = sum(All_iter_results5(:,1:test_num),2)/test_num;
ave_time5 = sum(All_time5(:,1:test_num),2)/test_num;
