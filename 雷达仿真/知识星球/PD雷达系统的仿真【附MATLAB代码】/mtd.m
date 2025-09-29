function [ S_Sigma_a S_Delta_a] =mtd(s_Sigma_rc,s_Delta_rc,Tr,Fs,Num_Tr_CPI )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
S_Sigma_a=cell(1,3);
S_Delta_a=cell(1,3);
for n_prf=1:length(Tr)
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  多普勒滤波（脉冲积累）  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Num_sample=round(Tr(n_prf)*Fs);
win_doppler=hamming(Num_Tr_CPI).';
S_Sigma_a{1,n_prf}=fftshift(fft(s_Sigma_rc{1,n_prf}(:,1:Num_Tr_CPI).*(ones(Num_sample,1)*win_doppler),[],2),2);
S_Delta_a{1,n_prf}=fftshift(fft(s_Delta_rc{1,n_prf}(:,1:Num_Tr_CPI).*(ones(Num_sample,1)*win_doppler),[],2),2);

% 将每个脉冲发射回波信号变换到多普勒域，取每行进行傅里叶变换。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

end

