%Freq. sel. Rayleigh block fading TDL model(uniformly spaced taps)
PDP_dBm = [0 -5 -10 -15 -20]; %Power delay profile values in dBm
L=5000; %number of channel realizations

N = length(PDP_dBm); %number of channel taps
PDP = 10.^(PDP_dBm/10); %PDP values in linear scale
a = sqrt(PDP); %path gains (tap coefficients a[n])

%Rayleigh random variables R0,R1,R2,...R{N-1} with unit avg power
%for each channel realization (block type fading)
R=1/sqrt(2)*(randn(L,N) + 1i*randn(L,N));%Rayleigh random variable
taps= repmat(a,L,1).*R; %combined tap weights = a[n]*R[n]
%Normalize taps for output with unit average power
normTaps = 1/sqrt(sum(PDP))*taps;

display('Average normalized power of each taps'); 
average_power = 20*log10(mean(abs(normTaps),1))
display('Overall path gain of the channel'); 
h_abs = sum(mean(abs(normTaps).^2,1))