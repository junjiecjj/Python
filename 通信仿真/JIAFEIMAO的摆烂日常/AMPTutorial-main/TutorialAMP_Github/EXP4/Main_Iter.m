clc;
clear all;

%% Parameters Setting
N=1024;                 %Dimension of x
M=512;                  %dimension of y
mes=0.90;               %damping factor
TestNum=1e0;            %Test Number
IterNum=20;             %Iteration number
snr=20;
rho=0.1;            
kappa=1e1;              %Condition number

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
   MAMP_MSE(:,kk)=MAMP_Detector(obj, Input);
   OAMP_MSE(:,kk)=OAMP_Detector(obj,Input);
   
   parfor_progress; 
end
parfor_progress(0);         % clear 

for ii=1:IterNum
    OAMP_MSE_mean(ii,1)=mean(OAMP_MSE(ii,:));
    MAMP_MSE_mean(ii,1)=mean(MAMP_MSE(ii,:));
end


figure(1)
iter=1:IterNum;

semilogy(iter,  OAMP_MSE_mean,'-ob')
hold on;
semilogy(iter,  MAMP_MSE_mean,'-*r')
hold on;

legend('OAMP','MAMP'); hold on;
xlabel('Iteration');
ylabel('NMSE');
