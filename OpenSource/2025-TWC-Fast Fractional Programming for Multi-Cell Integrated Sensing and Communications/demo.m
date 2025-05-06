%% demo run all
close all;clear; clc
%% parameters

L_T = [0.8,0.8]; % Location of Target

%% parameters for generating channels

Nt = 128; % # of Tx
Nr = 128; % # of Rx 
Factor = 1e6;
bandwidth = 10;
numBS = 7; L = numBS;


%% 
M = 4; % # Rxs of user in each cell (M in the paper)
K = 4; % data streams

sigma_c = 1e-8; sigma_r = 1e-7; T = 30; Pmax = 1e2;
num_SECTOR = 3; num_User_sector = 15;

%%
Q = num_SECTOR*num_User_sector; % num of users in each cell
numUser = Q*7; % total num of users
noise = 10^((-169-30)/10)*Factor;
maxPower = ones(1,numBS)*10^((-47-30)/10)*Factor;
mimoPattern = [Nt,M]; % [tx,rx]
numTone = 1;
weight = 1; mu = ones(Q,L)*weight;% weight for communication
beta1 = 1e-14;  % weight for sensing
eps = 1e-15;
MaxIter = 2*1e1; test_num = 1;
M_MaxIter = 5*MaxIter;

CFP_obj = zeros(MaxIter+1,test_num); CFP_rate = zeros(MaxIter+1,test_num); CFP_FI = zeros(MaxIter+1,test_num);  CFP_time = zeros(MaxIter,test_num);
NFP_obj = zeros(M_MaxIter+1,test_num); NFP_rate = zeros(M_MaxIter+1,test_num); NFP_FI = zeros(M_MaxIter+1,test_num);  NFP_time = zeros(M_MaxIter,test_num);
FFP_obj = zeros(M_MaxIter+1,test_num); FFP_rate = zeros(M_MaxIter+1,test_num); FFP_FI = zeros(M_MaxIter+1,test_num);  FFP_time = zeros(M_MaxIter,test_num);
for test = 1:test_num
    %% random generate channels and W
    [chn, distPathLoss,Location,A] = GenerateNetwork7_interference(bandwidth, numBS, numUser,mimoPattern,numTone);
    [W0] = generate_W(Nt,K,Pmax,L,Q);
    H = squeeze(chn);
    Location_difference = L_T-Location;
    % % 根据位置计算theta
    theta = atan(Location_difference(:,2)./Location_difference(:,1));
    [d_G] = generate_sensing_channel(Nt,Nr,L,theta);
    alpha = ones(L,1)*1e-3; 
    %% run 
     [CFP_obj(:,test),CFP_rate(:,test),CFP_FI(:,test),CFP_time(:,test)] = CFP_beamforming(MaxIter,sigma_c,sigma_r,Nt,Nr,L,Q,H,W0,T,alpha,M,K,A,d_G,mu,beta1,Pmax,eps);
    %% run QT
     [NFP_obj(:,test),NFP_rate(:,test),NFP_FI(:,test),NFP_time(:,test)] = NFP_beamforming(M_MaxIter,sigma_c,sigma_r,Nt,Nr,L,Q,H,W0,T,alpha,M,K,A,d_G,mu,beta1,Pmax,eps);
    %% run EQT
     [FFP_obj(:,test),FFP_rate(:,test),FFP_FI(:,test),FFP_time(:,test),W] = FFP_beamforming(M_MaxIter,sigma_c,sigma_r,Nt,Nr,L,Q,H,W0,T,alpha,M,K,A,d_G,mu,beta1,Pmax,eps);
end
