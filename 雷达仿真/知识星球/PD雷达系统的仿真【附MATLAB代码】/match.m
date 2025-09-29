function [s_Sigma_rc s_Delta_rc] = match(s_Sigma,s_Delta,Tr,Fs,K,Num_Tr_CPI)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  匹配滤波（脉冲压缩)  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%]


s_Sigma_rc=cell(1,3);
s_Delta_rc=cell(1,3);
for n_prf=1:length(Tr)
 s_S=s_Sigma{1,n_prf};
 s_D=s_Delta{1,n_prf};

Num_sample=round(Tr(n_prf)*Fs);
% 每个脉冲发射回波信号的长度
frange=((-Fs/2+Fs/(2*Num_sample)):Fs/Num_sample:(Fs/2-Fs/(2*Num_sample))).';
% 每个脉冲发射回波信号变换到频域后每个频域采样点对应的频率
Win=hamming(Num_sample);
% 窗函数（降低旁瓣）
H_match=exp(1i*pi*frange.^2/K).*Win;
% 匹配滤波函数
plot(H_match);
S_Sigma_r=fftshift(fft(s_S,[],1),1);
S_Delta_r=fftshift(fft(s_D,[],1),1);
% 将每个脉冲发射回波信号变换到频域
S_Sigma_r=S_Sigma_r.*(H_match*ones(1,Num_Tr_CPI));
S_Delta_r=S_Delta_r.*(H_match*ones(1,Num_Tr_CPI));
% 匹配滤波
s_Sigma_rc{1,n_prf}=ifft(ifftshift(S_Sigma_r,1),[],1);
s_Delta_rc{1,n_prf}=ifft(ifftshift(S_Delta_r,1),[],1);
% 将每个脉冲发射回波信号变换回时域
end

end

