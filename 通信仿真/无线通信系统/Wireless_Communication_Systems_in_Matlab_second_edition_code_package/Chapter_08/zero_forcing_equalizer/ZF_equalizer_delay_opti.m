%The test methodology takes the example given in 
%Yu Gong et al., Adaptive MMSE equalizer with Optimum tap-length and
%decision delay, sspd 2010 and try to get a similar plot as given in Figure
%3 of the journal paper
h_c=[-0.1 -0.3 0.4 1 0.4 0.3 -0.1]; %test channel
Ns=5:5:30; %sweep number of equalizer taps from 5 to 30
maxDelay=Ns(end)+length(h_c)-2; %max delay cannot exceed this value
err=zeros(length(Ns),maxDelay);
MSE=zeros(length(Ns),maxDelay);
optimalDelay=zeros(length(Ns),1);

i=1;j=1;
for N=Ns, %sweep number of equalizer taps
    for delays=0:1:N+length(h_c)-2, %sweep delays
        %compute MSE and  optimal delay for each combination
        [~,err(i,j),optimalDelay(i,1)]=zf_equalizer(h_c,N,delays);
        j=j+1;
    end
    i=i+1;j=1;
end
plot(0:1:N+length(h_c)-2,log10(err.')) %plot MSE in log scale
title('Error Vs equalizer delay for given channel and equalizer tap lengths');
xlabel('Equalizer delay');
ylabel('log_{10}[Error]');
%display optimal delays for each selected filter length N
%this will correspond with the bottom of the buckets displayed in the plot
disp('Optimal Delays for each N value ->')
disp(optimalDelay);