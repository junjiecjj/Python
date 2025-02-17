h_c1 = [0.04 -0.05 0.07 -0.21 -0.5 0.72 0.36 0.21 0.03 0.07]; %Channel A
h_c2 = [0.407 0.815 0.407]; %Channel B
h_c3 = [0.227 0.460 0.688 0.460 0.227]; %Channel C
%Compute and plot frequency responses
[H_c1,W1]=freqz(h_c1);
[H_c2,W2]=freqz(h_c2);
[H_c3,W3]=freqz(h_c3);
subplot(3,2,1);stem(h_c1);title('Channel A')
subplot(3,2,3);stem(h_c2);title('Channel B')
subplot(3,2,5);stem(h_c3);title('Channel C')
subplot(3,2,2);plot(W1,20*log10(abs(H_c1)/max(abs(H_c1))));
xlabel('Frequency \omega');ylabel('Amplitude (dB)');title('Channel A')
subplot(3,2,4);plot(W2,20*log10(abs(H_c2)/max(abs(H_c2))));
xlabel('Frequency \omega');ylabel('Amplitude (dB)');title('Channel B')
subplot(3,2,6);plot(W3,20*log10(abs(H_c3)/max(abs(H_c3))));
xlabel('Frequency \omega');ylabel('Amplitude (dB)');title('Channel C')