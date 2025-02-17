function [Xi,Ti] = poissonProcess(lambda,n)
%Generate inter-arrival times Xi (and hence arrival times Ti)
%from a Poisson point process with rate parameter lambda
%n denotes the number of events to generate
Xi = zeros(1,n+1);%include first event that occurs at t=0
Xi(2:n+1) = expRV(lambda,n); %n events with intensity lambda
Ti = sum(triu(repmat(Xi,[prod(size(Xi)) 1])')); %cumulative sum
end