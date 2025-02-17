%M-ary Partial Response Signaling (Discrete time equivalent model)
%with mod-M precoding. This is an ideal model with no noise added.
M=4; %Choose number of signaling levels for M-ary system
N=100000; %Number of symbols to transmit
a=randi([0,M-1],N,1) %random input stream of M-ary symbols
Q=[1 1]; % q0=1,q1=1, PR Class 1 Scheme (Duobinary coding)

x=zeros(size(a)); %output of precoder
D=zeros(length(Q),1); %memory elements in the precoder

%Precoder (Inverse filter of Q(z)) with Modulo-M operation
for k=1:length(a)
  x(k)=mod(a(k)-(D(2:end).*Q(2:end)),M);
  D(2)=x(k);
  if length(D)>2
   D(3:end)=D(2:end-1); %shift one symbol for next cycle
  end
end
disp(x); %display precoded output
bn=filter(Q,1,x)%Sampled received sequence-if desired,add noise here
acap=mod(bn,M) %modulo-M reduction at receiver
errors=sum(acap~=a) %Number of errors in the recovered sequence