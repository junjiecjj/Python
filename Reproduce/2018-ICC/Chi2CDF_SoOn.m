
clc;
clear all;
close all;

rng(42);

x = 0:0.2:15;
y = chi2cdf(x,  2);


figure(1);
plot(x,y)
xlabel('Observation')
ylabel('Cumulative Probability')

