function [s_Sigma_rc s_Delta_rc] = match(s_Sigma,s_Delta,Tr,Fs,K,Num_Tr_CPI)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  ƥ���˲�������ѹ��)  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%]


s_Sigma_rc=cell(1,3);
s_Delta_rc=cell(1,3);
for n_prf=1:length(Tr)
 s_S=s_Sigma{1,n_prf};
 s_D=s_Delta{1,n_prf};

Num_sample=round(Tr(n_prf)*Fs);
% ÿ�����巢��ز��źŵĳ���
frange=((-Fs/2+Fs/(2*Num_sample)):Fs/Num_sample:(Fs/2-Fs/(2*Num_sample))).';
% ÿ�����巢��ز��źű任��Ƶ���ÿ��Ƶ��������Ӧ��Ƶ��
Win=hamming(Num_sample);
% �������������԰꣩
H_match=exp(1i*pi*frange.^2/K).*Win;
% ƥ���˲�����
plot(H_match);
S_Sigma_r=fftshift(fft(s_S,[],1),1);
S_Delta_r=fftshift(fft(s_D,[],1),1);
% ��ÿ�����巢��ز��źű任��Ƶ��
S_Sigma_r=S_Sigma_r.*(H_match*ones(1,Num_Tr_CPI));
S_Delta_r=S_Delta_r.*(H_match*ones(1,Num_Tr_CPI));
% ƥ���˲�
s_Sigma_rc{1,n_prf}=ifft(ifftshift(S_Sigma_r,1),[],1);
s_Delta_rc{1,n_prf}=ifft(ifftshift(S_Delta_r,1),[],1);
% ��ÿ�����巢��ز��źű任��ʱ��
end

end

