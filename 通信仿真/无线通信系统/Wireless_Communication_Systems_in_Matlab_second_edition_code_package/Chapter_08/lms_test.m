%System identification using LMS algorithm
clearvars; clc;
N = 5; %length of the desired filter 
mu=0.1; %step size for LMS algorithm

r=randn(1,10000);%random input signal
h=randn(1,N)+1i*randn(1,N); %random complex system
a=conv(h,r);%reference signal
w=lms(N,mu,r,a);%designed filter using input signal and reference

disp('System impulse response (h):'); disp(h)
disp('LMS adapted filter (w): '); disp(w);