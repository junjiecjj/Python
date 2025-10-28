close all;
clear all;
clc;

%% This code illustrates the CFAR method for a signal with basic pulses.

%% Variables
Fs = 1e8;  % Sampling frequency
dt = 1/Fs; % Sampling period
pulse_length = 1e-6; % Pulse length
Time = 1e-3; % Time duration
t = 0:dt:Time-dt; % Time scale

%% Signal construction and matched filtering
reference_pulse = 1*(t>=0 & t<=2*pulse_length); % Reference pulse
matched_filter = reference_pulse(end:-1:1).'; % Matched filter 
signal = 1*(t>=5*pulse_length & t<=6*pulse_length) + 1*(t>=100*pulse_length & t<=101*pulse_length) + 0.2*randn(1,length(t)); % The constructed signal with noise
signal_matched_filtered = abs(conv(signal,matched_filter)); % Matched filtering
signal_matched_filtered = signal_matched_filtered(round(length(signal_matched_filtered)/2):end); % Selecting the useful part

%% CFAR
guard_cell_number = 800; % Guard cell number
reference_cell_number = 50; % One sided reference cell number
pfa = 1e-5; % Probability of false alarm
N = 2*reference_cell_number; % Number of reference cells
alpha = N*(pfa^(-1/N)-1); % CFAR coefficient
cfar_vec = [ones(1,reference_cell_number),zeros(1,guard_cell_number),ones(1,reference_cell_number)]; % CFAR window
cfar_threshold = alpha*conv(signal_matched_filtered,cfar_vec,'same')/N; % CFAR threshold

%% Plots
figure;
tiledlayout(2,1);
ax1 = nexttile;
plot(t,cfar_threshold)
hold on 
plot(t,signal_matched_filtered)
title("CFAR Detection Illustration")
xlabel("Time(s)");
ylabel("Voltage(V)")
legend("CFAR Threshold","Signal")
ax2 = nexttile;
plot(t,20*log10(cfar_threshold));
hold on;
plot(t,20*log10(signal_matched_filtered))
title("CFAR Detection Illustration in dB")
xlabel("Time(s)");
ylabel("Voltage(dBV)")
legend("CFAR Threshold","Signal")
