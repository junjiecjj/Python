clc;
clear all;

%% Parameters Setting
N=1024;                 %Dimension of x
M=512;                  %dimension of y
mes=0.90;               %damping factor
TestNum=1e1;            %Iteration numbers
IterNum=20;
snr=20;
rho=0.1;
kappa=1e0;              %Condition number

%% Load parameters
Input.N=N;
Input.M=M;
Input.mes=mes;
Input.IterNum=IterNum;
Input.nuw=10^(-snr/10);
Input.rho=rho;
Input.kappa=kappa;

%% Array setting
parfor_progress(TestNum);   % Counter
for kk=1:TestNum
    
   obj=MIMO_system(Input);
   AMP_MSE(:,kk)=AMP_Detector(obj,Input);
   OAMP_MSE(:,kk)=OAMP_Detector(obj,Input);
   OAMPse(:,kk)=OAMP_SE(Input,obj);
   parfor_progress; 
end
parfor_progress(0);         % clear 

for ii=1:IterNum
    AMP_MSE_mean(ii,1)=mean(AMP_MSE(ii,:));
    OAMP_MSE_mean(ii,1)=mean(OAMP_MSE(ii,:));
    OAMP_SE_mean(ii,1)=mean(OAMPse(ii,:));
end

AMP_SE_mean =AMP_SE(Input);

figure(1)
iter=1:IterNum;

semilogy(iter,  AMP_MSE_mean,'-r')
hold on;
semilogy(iter,  AMP_SE_mean,'ob')
hold on;
semilogy(iter,  OAMP_MSE_mean,'-k')
hold on;
semilogy(iter,  OAMP_SE_mean,'sm')
hold on;


legend('AMP','AMP SE','OAMP', 'OAMP SE'); hold on;
xlabel('Iteration');
ylabel('NMSE');
