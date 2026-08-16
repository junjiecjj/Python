
clc;
clear all;
close all;

rng(42);

x = 0:0.2:50;
y = chi2pdf(x, 2);

figure(1);
plot(x,y)
xlabel('Observation')
ylabel('Probability Density')



x = 0:0.2:50;
y = chi2cdf(x, 2);

figure(2);
plot(x,y)
xlabel('Observation')
ylabel('Cumulative Probability')
