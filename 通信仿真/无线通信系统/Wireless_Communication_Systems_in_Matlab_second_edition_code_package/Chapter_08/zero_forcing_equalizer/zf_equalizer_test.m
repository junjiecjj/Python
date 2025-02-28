clearvars; clc;
%System parameters
nSamp=5; %Number of samples per symbol determines baud rate Tsym
Fs=100; %Sampling Frequency of the system
Ts=1/Fs; %Sampling time
Tsym=nSamp*Ts; %symbol time period

%Define transfer function of the channel
k=6; %define limits for computing channel response
N0 = 0; %Standard deviation of AWGN channel noise
t = -k*Tsym:Ts:k*Tsym; %time base defined till +/-kTsym
h_t = 1./(1+(t/Tsym).^2);%channel model, replace with your own model
h_t = h_t + N0*randn(1,length(h_t));%add Noise to the channel response
h_k = h_t(1:nSamp:end);%downsampling to represent symbol rate sampler
t_inst=t(1:nSamp:end); %symbol sampling instants

figure; plot(t,h_t); hold on; %channel response at all sampling instants
stem(t_inst,h_k,'r'); %channel response at symbol sampling instants
legend('continuous-time model','discrete-time model');
title('Channel impulse response'); xlabel('Time (s)');ylabel('Amplitude');

%Equalizer Design Parameters
N = 14; %Desired number of taps for equalizer filter
delay = 11;

%design zero-forcing equalizer for given channel and get tap weights
%and filter the input through the equalizer find equalizer co-effs for given CIR
[w,error,k0]=zf_equalizer(h_k,N,delay);
%[w,error,k0]=zf_equalizer(h,N,); %Try this delay optimized equalizer
r_k=h_k; %Test the equalizer with the sampled channel response as input
d_k=conv(w,r_k); %filter input through the eq
h_sys=conv(w,h_k); %overall effect of channel and equalizer

disp(['ZF equalizer design: N=', num2str(N), ...
    ' Delay=',num2str(delay), ' error=', num2str(error)]);
disp('ZF equalizer weights:'); disp(w);

%Frequency response of channel,equalizer & overall system
[H_F,Omega_1]=freqz(h_k);     %frequency response of channel
[W,Omega_2]=freqz(w);   %frequency response of equalizer
[H_sys,Omega_3]=freqz(h_sys); %frequency response of overall system

figure; 
plot(Omega_1/pi,20*log(abs(H_F)/max(abs(H_F))),'g'); hold on;
plot(Omega_2/pi,20*log(abs(W)/max(abs(W))),'r'); 
plot(Omega_3/pi,20*log(abs(H_sys)/max(abs(H_sys))),'k');
legend('channel','ZF equalizer','overall system');
title('Frequency response');ylabel('Magnitude(dB)');
xlabel('Normalized frequency(x \pi rad/sample)');

figure; %Plot equalizer input and output(time-domain response)
subplot(2,1,1); stem(0:1:length(r_k)-1,r_k); 
title('Equalizer input'); xlabel('Samples'); ylabel('Amplitude');
subplot(2,1,2); stem(0:1:length(d_k)-1,d_k);
title(['Equalizer output- N=', num2str(N), ...
    ' Delay=',num2str(delay), ' Error=', num2str(error)]); 
xlabel('Samples'); ylabel('Amplitude');