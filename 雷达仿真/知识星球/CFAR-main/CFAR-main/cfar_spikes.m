close all;
clear all;
clc;

%% This code illustrates the CFAR method for a signal with peaks in some loctions.

%% Variables
Fs = 1e8;  % Sampling frequency
dt = 1/Fs; % Sampling period
Time = 1e-4; % Time duration
t = 0:dt:Time-dt; % Time scale

%% Constructing the signal with spikes in some locations.
signal = randn(1,length(t)); % Random signal
signal([400 800 4000]) = [10 15 9]; % Putting peaks

%% CFAR
guard_cell_number = 3; % Number of guard cells
reference_cell_number = 50; % Number of one sided reference cells
cfar_vec = [ones(1,reference_cell_number),zeros(1,guard_cell_number),ones(1,reference_cell_number)]; % CFAR vector
cfar_vec = cfar_vec/sum(cfar_vec);
pfa = 1e-5; % Probability of false alarm
N = 2*reference_cell_number; % Numbe rof reference cells
alpha = N*(pfa^(-1/N)-1); % CFAR coefficient
cfar_threshold = alpha*conv(abs(signal),cfar_vec,'same'); % CFAR threshold

%% Plots
figure;
tiledlayout(2,1);
ax1 = nexttile;
plot(t,cfar_threshold)
hold on 
plot(t,abs(signal))
title("CFAR Detection Illustration")
xlabel("Time(s)");
ylabel("Voltage(V)")
legend("CFAR Threshold","Signal")
ax2 = nexttile;
plot(t,20*log10(cfar_threshold));
hold on;
plot(t,20*log10(abs(signal)))
title("CFAR Detection Illustration in dB")
xlabel("Time(s)");
ylabel("Voltage(dBV)")
legend("CFAR Threshold","Signal")