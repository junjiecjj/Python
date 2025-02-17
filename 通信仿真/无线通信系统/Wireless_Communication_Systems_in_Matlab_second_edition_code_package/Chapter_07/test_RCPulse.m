Tsym=1; %Symbol duration in seconds
L=10; % oversampling rate, each symbol contains L samples
Nsym = 80; %filter span in symbol durations
alphas=[0 0.3 0.5 1];%RC roll-off factors - valid range 0 to 1
Fs=L/Tsym;%sampling frequency

lineColors=['b','r','g','k','c']; i=1;legendString=cell(1,4);

for alpha=alphas %loop for various alpha values    
 [rcPulse,t]=raisedCosineFunction(alpha,L,Nsym); %RC Pulse
 subplot(1,2,1); t=Tsym*t; %translate time base for given duration
 plot(t,rcPulse,lineColors(i));hold on; %plot time domain view
 
 [vals,f]=freqDomainView(rcPulse,Fs,'double');%See Chapter 1
 subplot(1,2,2);
 plot(f,abs(vals)/abs(vals(length(vals)/2+1)),lineColors(i)); 
 hold on;legendString{i}=strcat('\alpha =',num2str(alpha) );i=i+1;  
end
subplot(1,2,1);title('Raised Cosine pulse'); legend(legendString);
subplot(1,2,2);title('Frequency response');legend(legendString);