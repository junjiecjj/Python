lambda = 5; n=500;%intensity and number of events
[Xi,Ti] = poissonProcess(lambda,n);%Process model
%Xi-interarrival time, Ta - actual arrival times

Delta=min(Xi(2:n+1)); %shortest interarrival time
t = 0:Delta:sum(Xi); %construct time base
N = zeros(size(Ti)); %Poisson process count N(t)
for i=1:n+1,
    N(t>=Ti(i))=i; %counting process
end

subplot(1,2,1);plot(t, N); xlabel('time'); ylabel('N(t)');
title('Poisson process');legend('N(t)','Events');
hold on; stem(Ti,zeros(size(Ti))+0.5,'^');%mark events 
axis([0 Ti(10) 0 10]); %zoom to first 10 events
subplot(1,2,2);histogram(Xi,'Normalization','pdf');
title('PDF of interarrival times');
xlabel('Interarrival time');ylabel('frequency');