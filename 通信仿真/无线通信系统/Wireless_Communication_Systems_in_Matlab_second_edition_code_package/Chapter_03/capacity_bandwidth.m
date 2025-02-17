PN = 1e6; %fix the SNR
B = (1:0.5e6:30e6); %vary the bandwidth
C = B.*log2(1+PN./B);
plot(B/1e6,C/1e6);
xlabel('Bandwidth (MHz)');ylabel('Capacity - C(B)- Mbps');
limit = PN*log2(exp(1))/1e6;
xL = get(gca,'XLim');
line(xL,[limit limit],'Color','r','LineStyle','--');