clc;
clear all;

%% Parameters Setting
N=1024;
M=512;
mes=0.98;
TestNum=1e2;
IterNum=30;
rho=0.05;
snr=15;

%% Load Parameters
Input.N=N;
Input.M=M;
Input.IterNum=IterNum;
Input.rho=rho;
Input.mes=mes;
Input.nuw=10^(-snr/10);


parfor_progress(TestNum);
for Index=1:TestNum
    obj=MIMO_system(Input);
    AMP_MSE(:,Index)=AMP(obj,Input);
    parfor_progress;   
end
parfor_progress(0); 

for kk=1:IterNum
    AMP_MSE_mean(kk)=mean(AMP_MSE(kk,:));
end

MSE =AMP_SE(Input);

figure(1)
Dis=1:IterNum;
semilogy(Dis,  AMP_MSE_mean, '-b');   hold on;
semilogy(Dis,  MSE, 'or');   hold on;
legend('AMP', 'SE'); 
hold on

xlabel('SNR (dB)');
ylabel('MSE');
