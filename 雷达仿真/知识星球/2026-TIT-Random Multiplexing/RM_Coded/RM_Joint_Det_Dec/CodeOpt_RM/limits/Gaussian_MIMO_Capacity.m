function C = Gaussian_MIMO_Capacity(beta_in, sigma2_in)
% Gaussian Capacity calculation of y = Hx + n;
% H: NrxNu, hij: N(0,1/Nr);
% beta = Nu/Nr;

%clc; clear all;
%beta =[0.5];%[0.1 0.5 1 2 10];%[1:2:10];
%snr_dB = 0;%-10:50; 
%snr = 10.^(snr_dB/10);
beta = beta_in;
snr  = 1 ./ sigma2_in;
%snr_dB = 10 .*log10(snr);
C = C_Gau(beta,snr);
% save data;
% %%
% load data;
% for i=1:length(beta)
% subplot(1,length(beta),i);
% plot(snr_dB,C(i,:),'r-','linewidth',1.5);
% legend('Capacity','location','northwest')
% title(['beta=' num2str(beta(i))]);
% xlabel('SNR (dB)');
% ylabel('Rate (bits)');
% end

end

function C = C_Gau(beta,snr)
   
C =zeros(length(beta),length(snr));
    for i=1:length(beta)
        snr_equ = snr;         %equivalent snr   /alpha(i);
        C(i,:) = beta(i)*log(1+snr_equ-0.25*F_fun(snr_equ,beta(i))) ...
                 + log(1+beta(i)*snr_equ-0.25*F_fun(snr_equ,beta(i))) - 0.25./snr_equ.*F_fun(snr_equ,beta(i));
        C(i,:) = C(i,:)/beta(i)/log(2);      % per transmission, per receive dont /beta(i)
    end

end

function y = F_fun(x,z)
y =( sqrt(x.*(1+sqrt(z)).^2+1) - sqrt(x.*(1-sqrt(z)).^2+1) ).^2;
end
 




