
clc;
clear all;


%% Parameters
N=1024;
M=512;
rho=0.05;
snr_dB=50;
TestNum=1e1;
IterNum=3*1e2;
lambda=0.05;


%% Load parameters
Input.N=N;
Input.M=M;
Input.rho=rho;
Input.nuw=10^(-snr_dB/10);
Input.IterNum=IterNum;
Input.lambda=lambda;


parfor_progress(TestNum);
for ii=1:TestNum
    obj=system_model(Input);
    [MSE_AMP(:,ii),  error_AMP(:,ii)] = AMP_Lasso(Input, obj);
    [MSE_ISTA(:,ii), error_ISTA(:,ii)]= ISTA_Lasso(Input, obj);
    [MSE_FISTA(:,ii), error_FISTA(:,ii)]= FISTA_Lasso(Input, obj);
     parfor_progress;   
end
parfor_progress(0);

for index=1:IterNum
    MSE_AMP_mean(index,1)=mean(MSE_AMP(index,:));
    MSE_ISTA_mean(index,1)=mean(MSE_ISTA(index,:));
    MSE_FISTA_mean(index,1)=mean(MSE_FISTA(index,:));
end


 
figure (1)
semilogy(MSE_AMP_mean,'-ob');
hold on
semilogy(MSE_FISTA_mean,'-sr');
hold on
semilogy(MSE_ISTA_mean,'-*k');
hold on
legend('AMP', 'FISTA', 'ISTA');
hold on