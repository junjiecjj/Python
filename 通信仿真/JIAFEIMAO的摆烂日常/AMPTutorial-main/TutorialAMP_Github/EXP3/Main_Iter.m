clc;
clear all;

%% Parameters Setting
N=1024;                 %Dimension of x
M=512;                  %dimension of y
mes=0.90;               %damping factor
TestNum=1e1;            %Test number 
IterNum=20;             %Iteration number
snr=10;
rho=0.05;
kappa=1e0;               %condition number

%% Load parameters
Input.N=N;
Input.M=M;
Input.mes=mes;
Input.IterNum=IterNum;
Input.nuw=10^(-snr/10);
Input.rho=rho;
Input.kappa=kappa;


parfor_progress(TestNum);   % Counter
for kk=1:TestNum
    
   obj=MIMO_system(Input);
   AMP_MSE(:,kk)=AMP_Detector(obj,Input);
   OAMP_MSE(:,kk)=OAMP_Detector(obj,Input);
   parfor_progress; 
end
parfor_progress(0);         % clear 

for ii=1:IterNum
    AMP_MSE_mean(ii,1)=mean(AMP_MSE(ii,:));
    OAMP_MSE_mean(ii,1)=mean(OAMP_MSE(ii,:));
end

figure(1)
iter=1:IterNum;

semilogy(iter,  AMP_MSE_mean,'-ro')
hold on;

semilogy(iter,  OAMP_MSE_mean,'-k*')
hold on;


legend('AMP','OAMP'); hold on;
xlabel('iteration');
ylabel('NMSE');
