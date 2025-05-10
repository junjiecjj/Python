clear;
clc;
close all;

vec = -10:0.5:30;%信噪比，SNR
vec_linear = 10.^(vec/10);
Shannon_limit1 = log2(1+vec_linear);

plot(vec,Shannon_limit1,'-hr');
hold on;
grid on;

legend( 'Shannon limit');

