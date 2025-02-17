clearvars; clc
nBits=100000; %number of bits to transmit
errorProbs = 0:0.1:1; %erasure probabilities of BEC to simulate
C = zeros(1,length(errorProbs)); %to store capacity for each point

j=1;
for e=errorProbs,%for each error probability
    x = rand(1,nBits)<0.5; %uniform source bits of 1's and 0's
    y = bec_channel(x,e); %process the bits through BEC    
    
    pye = sum(y==-1)/nBits ;%py(y=-1), prob of erasure from data
    C(j) = 1-pye; %capacity    
    j=j+1;
end
plot(errorProbs,C); title('Capacity over Binary Erasure Channel');
xlabel('Erasure probability - e'); 
ylabel('Capacity (bits/channel use)');