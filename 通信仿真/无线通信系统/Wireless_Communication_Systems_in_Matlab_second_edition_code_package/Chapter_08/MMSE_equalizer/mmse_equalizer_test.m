clearvars clc;
%System parameters
nSamp=5; %Number of samples per symbol determines baud rate Tsym
Fs=100; %Sampling Frequency of the system
Ts=1/Fs; %Sampling period
Tsym=nSamp*Ts;

%Define transfer function of the channel
k=6; %define limits for computing channel response
N0 = 0.1; %Standard deviation of AWGN noise
t = -k*Tsym:Ts:k*Tsym; %time base defined till +/-kTsym
h_t = 1./(1+(t/Tsym).^2); %channel model, replace with your own model
h_t = h_t + N0*randn(1,length(h_t)); %add Noise to the channel response
h_k= h_t(1:nSamp:end);%downsampling to represent symbol rate sampler
t_k=t(1:nSamp:end); %symbol sampling instants

figure;plot(t,h_t);hold on;%channel response at all sampling instants
stem(t_k,h_k,'r'); %channel response at symbol sampling instants
legend('continuous-time model','discrete-time model');
title('Channel impulse response'); 
xlabel('Time (s)');ylabel('Amplitude');

%Equalizer Design Parameters
nTaps = 14; %Desired number of taps for equalizer filter

%design DELAY OPTIMIZED MMSE eq. for given channel, get tap weights
%and filter the input through the equalizer
noiseVariance = N0^2; %noise variance
snr = 10*log10(1/N0); %convert to SNR (assume var(signal) = 1)
[w,mse,optDelay]=mmse_equalizer(h_k,snr,nTaps); %find eq. co-effs

r_k=h_k; %Test the equalizer with the channel response as input
d_k=conv(w,r_k); %filter input through the equalizer
h_sys=conv(w,h_k); %overall effect of channel and equalizer

disp(['MMSE equalizer design: N=', num2str(nTaps),' delay=',num2str(optDelay)])
disp('MMSE equalizer weights:'); disp(w)

figure; subplot(1,2,1); stem(0:1:length(r_k)-1,r_k);
xlabel('Samples'); ylabel('Amplitude');title('Equalizer input');
subplot(1,2,2); stem(0:1:length(d_k)-1,d_k);
xlabel('Samples'); ylabel('Amplitude');
title(['Equalizer output- N=', num2str(nTaps),' delay=',num2str(optDelay)]);