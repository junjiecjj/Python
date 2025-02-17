function [S,f,r,tau] = doppler_psd_acf(TYPE,T_s,f_max,sigma_0_2,N)
%Jakes or Gaussian PSD and their auto-correlation functions (ACF)
%TYPE='JAKES' or 'GAUSSIAN'
%Ts = sampling frequency in seconds
%f_max = maximum doppler frequency in Hz
%sigma_0_2 variance of gaussian processes (sigma_0^2)
%N = number of samples in ACF plot
%EXAMPLE: doppler_psd_acf('JAKES',0.001,91,1,1024)

tau = T_s*(-N/2:N/2); %time intervals for ACF computation
if strcmpi(TYPE,'Jakes'),
    r=sigma_0_2*besselj(0,2*pi*f_max*tau);%JAKES ACF
elseif strcmpi(TYPE,'Gaussian'),
    f_c = sqrt(log(2))*f_max; %3-dB cut-off frequency
    r=sigma_0_2*exp(-(pi*f_c/sqrt(log(2)).*tau).^2);%Gaussian ACF
else
    error('Invalid PSD TYPE specified')
end
%Power spectral density using FFT of ACF
[S,f] = freqDomainView(r,1/T_s,'double');%chapter 1, section 1.3.4

subplot(1,2,1); plot(f,abs(S)); xlim([-200,200]);
title([TYPE,' PSD']);xlabel('f(Hz)');ylabel('S_{\mu_i\mu_i}(f)');
subplot(1,2,2); plot(tau,r);axis tight; xlim([-0.1,0.1]);
title('Auto-correlation fn.');xlabel('\tau(s)');ylabel('r_{\mu_i\mu_i}(\tau)');
end