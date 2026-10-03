clear all;
%%%  parameters' definition
c=3e+8;										% speed of light
pi=3.1415926; 
j=sqrt(-1);	

Tp=1.e-6; 								% transmitted pulse width      
fc=1.e+9;	 						   	% carrier frequency 
Br=50.e+6;              % transmitted bandwidth
Fs=200.e+6;             % A/D sample rate
kr=Br/Tp;               % range chirp rate

Nr=Tp*Fs;
Ni=1:Nr;
tr=(Ni-Nr/2)*Tp/Nr;

%===============================
%Generate a Chirp pulse
%===============================

sig = exp(j*pi*kr*(tr).^2);

figure; 
subplot(2,3,1); plot(real(sig));
subplot(2,3,2); plot(imag(sig));

%===============================
% Spectrum of this Chirp pulse
%===============================

sig_spec=fftshift(fft(sig));
subplot(2,3,3); ; plot(abs(sig_spec));

%===============================
%Match filtering in time domain
%===============================

mf_sig = conj(fliplr(sig));
subplot(2,3,4);  plot(real(mf_sig));
subplot(2,3,5);  plot(imag(mf_sig));

mf_out = conv(sig, mf_sig);
subplot(2,3,6);  plot(abs(mf_out(Nr/2+1:Nr*3/2)));

pause;
%=====================================
%Match filtering in frequency domain
%=====================================

fr=(Ni-Nr/2)*Fs/Nr;
mf_spec=exp(j*pi*fr.^2/kr);
mf_spec_out=ifft(fftshift(sig_spec.*mf_spec));

figure; 
subplot(1,2,1); plot(abs(mf_spec_out));

sig_un=20*log10(abs(mf_spec_out)/max(abs(mf_spec_out)));
for i=1 :Nr
    if (sig_un(i)<-40 )
        sig_un(i)=-40;
    end 
end
subplot(1,2,2); plot(sig_un);

% D:\ºÏ³É¿×¾¶À×´ï\chirp_sig.m 