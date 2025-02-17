%Impulse response and Frequency response of PR signaling shemes
clearvars; clc;
L = 50; %oversample factor Tsym/Ts
Nsym = 8; %PRS filter span in symbols

QD_arr = cell(8,1); %place holder for 8 different PR co-effs Q(D)
QD_arr{1}=[1 1]; %PR Class I Duobinary scheme
QD_arr{2}=[1 -1]; %PR Class I Dicode channel scheme
QD_arr{3}=[1 2 1]; %PR Class II
QD_arr{4}=[2 1 -1]; %PR Class III
QD_arr{5}=[1 0 -1]; %PR Class IV (Modified Duobinary)
QD_arr{6}=[1 1 -1 -1]; %EPR4 (Enhanced Class IV)
QD_arr{7}=[1 2 0 -2 -1]; %E2PR4 (Enhanced EPR4)
QD_arr{8}=[1 0 -2 0 1]; %PR Class V
A=1;%filter co-effs in Z-domain(denominator) for any FIR type filter
titles={'PR1 Duobinary','PR1 Dicode','PR Class II','PR Class III'...
    ,'PR4 Modified Duobinary','EPR4','E2PR4','PR Class V'};

for i=1:8 %loop for each PR scheme
 %--------Impulse response------------------
 Q = QD_arr{i}; %tap values for Q(f) filter    
 %construct PR signals as cascaded combination of Q(f) and G(f)
 [b,t]=PRSignaling(Q,L,Nsym);%impulse response selected PR scheme
 subplot(1,2,1);plot(t,b); hold on;
 stem(t(1:L:end),b(1:L:end),'r'); %Highlight symbol sample instants
 grid on; title([titles{i} '- b(t)']);
 xlabel('t/T_{sym}'); ylabel('b(t)'); axis tight;hold off;
 
 %--------Frequency response---------------
 [H,W]=freqz(Q,A,1024,'whole');%Frequency response
 H=[H(length(H)/2+1:end); H(1:length(H)/2)];%for double sided plot
 response=abs(H);
 norm_response=response/max(response);%Normalize response
 norm_frequency= W/max(W)-0.5;%Normalized frequency to -0.5 to 0.5
 subplot(1,2,2); plot(norm_frequency,norm_response,'b');
 grid on; axis tight; title([titles{i} '- Frequency response Q(D)'])
 xlabel('f/F_{sym}');ylabel('|Q(D)|')   
 
 pause %pause for user response
end