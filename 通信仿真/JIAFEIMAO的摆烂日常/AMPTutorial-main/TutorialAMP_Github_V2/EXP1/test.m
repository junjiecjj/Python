clc;clear all;close all;



nBits=10^5; %number of bits to transmit 
errorProbs = 0:0.1:0.6; %cross-over probabilities of BSC to simulate 
C = zeros(1,length(errorProbs)); %to store capacity for each point
 
j=1; 
for e=errorProbs, %for each error probability 
    x = rand(1,nBits)<0.5; %uniform source bits of 1's and 0's 
    y = bsc_channel(x,e); %process the bits through BSC
 
    Y = repmat(y,2,1);%construct 2xL matrix from y 
    X = repmat([0;1],1,nBits);%construct 2xL matrix from [0;1] %probabilities with respect to each input[0,1] 

    prob = (Y==X)*(1-e)+(Y~=X)*e; %BSC p(y/x) equation 
    prob = max(prob,realmin);%this used to avoid NAN in computation 
    p = prob./(ones(2,1)*(sum(prob)));%normalize probabilities 
    HYX = mean(-sum(p.*log2(p))); %senders uncertainity

    py0 = sum(y==0)/nBits; %py(y=0) 
    py1 =sum(y==1)/nBits; %py(y=1) 
    HY = -py0*log2(py0) - py1*log2(py1);%avg info content of output 
    C(j) = HY-HYX; %capacity 
    j=j+1;

end 
plot(errorProbs,C); title('Capacity over Binary Symmetric Channel');
xlabel('Cross-over probability - e'); 
ylabel('Capacity (bits/channel use)');


function y = bsc_channel(x,e) %Process an input vector x through a Binary Symmetric Channel with %error probability e. 
y = x; %copy the input vector 
errors = rand(1,length(x))<e; %Bernoulli trials with probability e 
y(errors) = 1 - y(errors); %flip the bits
end