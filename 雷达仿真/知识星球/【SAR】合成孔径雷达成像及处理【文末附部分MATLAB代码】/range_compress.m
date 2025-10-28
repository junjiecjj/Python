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
%Chirp pulse echo from point A
%===============================

sig_point0 = exp(j*pi*kr*(tr).^2);

%===============================
%Chirp pulse echo from point B
%===============================

dr=3;  dr_t=2*dr/c;  dN=dr_t*Fs;
sig_point1 = exp(j*pi*kr*(tr-dr_t).^2);

sig_0 = zeros(1, 3*Nr); sig_1 = sig_0; sum_sig = sig_0;

sig_0(Nr+1:2*Nr)=sig_point0;
sig_1(Nr+dN+1:2*Nr+dN)=sig_point1;

%===============================
% Summary echo signal of A and B
%===============================

sum_sig = sig_0 + sig_1;

figure;
subplot(4,1,1); plot(real(sig_0));
subplot(4,1,2); plot(real(sig_1));
subplot(4,1,3); plot(real(sum_sig));

%=====================================
% Range compression with match filter
%=====================================

mf_sig = conj(fliplr(sig_point0));
mf_out = conv(sum_sig, mf_sig);
Ns=Nr/2+1; Ne=3*Nr+Nr/2;
subplot(4,1,4); plot(abs(mf_out(Ns:Ne)));

 