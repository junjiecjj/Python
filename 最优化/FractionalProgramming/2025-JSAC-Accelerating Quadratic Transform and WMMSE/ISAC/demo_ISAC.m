%% demo average 200 run of GQT
clear
clc
test_num = 1;
max_iter = 20;
sigma1 = 1e-8; sigma2 = 1e-8;
theta = 1/4*pi;
Nr = 72; M = 64; N = 2;
P_max = ones(2,1)*1e2;
iter_results = zeros(max_iter+1,1); error = zeros(max_iter,1); time = zeros(max_iter,1);
sigma_s = 1e-8;
stop_time = 50;
L_B= [0,0; 250,0]; % location of BSs
L_U = [-10,100;350,-100]; % location of Users
L_T = [200,200]; % locations of target

different_w = [1e5]; 
%% GQT
All_iter_results1 = zeros(10*max_iter+1,test_num);
All_time1 = zeros(10*max_iter,test_num);
%% EQT
All_iter_results2 = zeros(10*max_iter+1,test_num);
All_time2 = zeros(10*max_iter,test_num);
%% CQT
All_iter_results3 = zeros(max_iter+1,test_num);
All_time3 = zeros(max_iter,test_num);

FI_1 = zeros(test_num,1); FI_2 = zeros(test_num,1); FI_3 = zeros(test_num,1);
S1_G = zeros(test_num,1); S1_E = zeros(test_num,1); S1_C = zeros(test_num,1); % 分别对应各个算法的SINR1
S2_G = zeros(test_num,1); S2_E = zeros(test_num,1); S2_C = zeros(test_num,1); % 分别对应各个算法的SINR2

%% generate A 
for m = 1:1
    a{m} = zeros(M(m),1); a_prime{m} = zeros(M(m),1);
    b{m} = zeros(Nr(m),1); b_prime{m} = zeros(Nr(m),1);
    for i =1:M(m)
        a{m}(i) = exp(-1i*pi*sin(theta(m))*(i-1));
        a_prime{m}(i) = -1i*pi*(i-1)*a{m}(i)*cos(theta(m));
    end
    for i = 1:Nr(m)
        b{m}(i) = exp(-1i*pi*sin(theta(m))*(i-1));
        b_prime{m}(i) = -1i*pi*(i-1)*b{m}(i)*cos(theta(m));
    end
    A = b_prime{m}*a{m}.'+b{m}*a_prime{m}.'; % \dot A
end

for i = 1:size(different_w,2)
    w = different_w(i);
    for test = 1:test_num
        [H,alpha,G] = generate_channel(L_B,L_U,L_T,P_max,M,N,Nr);
        H11 = H{1,1}; H12 = H{1,2}; H22 = H{2,2}; H21 = H{2,1};
        [v1,v2] = generate_V(M,P_max); % initialze v
        % CQT
        [All_iter_results1(:,test),  All_time1(:,test),FI_1(test),S1_G(test),S2_G(test)] = one_GQT(10*max_iter,sigma1,sigma2,H11,H12,H21,H22,G,alpha,Nr,M,N,w,sigma_s,P_max,A,v1,v2);
        % NQT
        [All_iter_results2(:,test),  All_time2(:,test),FI_2(test),S1_E(test),S2_E(test)] = one_NQT(10*max_iter,sigma1,sigma2,H11,H12,H21,H22,G,alpha,Nr,M,N,w,sigma_s,P_max,A,v1,v2);
        % CQT
        [All_iter_results3(:,test),  All_time3(:,test),FI_3(test),S1_C(test),S2_C(test)] = one_CQT(max_iter,sigma1,sigma2,H11,H12,H21,H22,G,alpha,Nr,M,N,w,sigma_s,P_max,A,v1,v2);
    end
end
