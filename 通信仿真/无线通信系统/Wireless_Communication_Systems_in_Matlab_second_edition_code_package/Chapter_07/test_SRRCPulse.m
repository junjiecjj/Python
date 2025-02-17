Tsym=1; %Symbol duration in seconds
L=10; % oversampling rate, each symbol contains L samples
Nsym = 80; %filter span in symbol durations
betas=[0 0.22 0.5 1];%root raised-cosine roll-off factors
Fs=L/Tsym;%sampling frequency
lineColors=['b','r','g','k','c']; i=1;legendString=cell(1,4);

for beta=betas %loop for various alpha values    
 [srrcPulseAtTx,t]=srrcFunction(beta,L,Nsym); %SRRC Filter at Tx
 srrcPulseAtRx = srrcPulseAtTx;%Using the same filter at Rx
 %Combined response matters as it hits 0 at desired sampling instants
 combinedResponse = conv(srrcPulseAtTx,srrcPulseAtRx,'same');
 
 subplot(1,2,1); t=Tsym*t; %translate time base & normalize reponse
 plot(t,combinedResponse/max(combinedResponse),lineColors(i));
 hold on;
 %See Chapter 1 for the function 'freqDomainView'
 [vals,F]=freqDomainView(srrcPulseAtTx,Fs,'double');
 subplot(1,2,2);
 plot(F,abs(vals)/abs(vals(length(vals)/2+1)),lineColors(i)); 
 hold on;legendString{i}=strcat('\beta =',num2str(beta) );i=i+1;  
end
subplot(1,2,1);
title('Combined response of SRRC filters'); legend(legendString);
subplot(1,2,2); 
title('Frequency response (at Tx/Rx only)');legend(legendString);