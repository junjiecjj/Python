close all;
clear all;
clc;

%% This code illustrates a more realistic radar scenerio for CFAR usage with LFM pulse signal.

%% Variables
Fs = 1e7;  % Sampling frequency
dt = 1/Fs; % Sampling period
Range = 1000; % Range of the target
antenna_gain = 30; % Antenna gain in dB
rcs = 0.1; % Radar cross section
light_speed = physconst("LightSpeed"); % Light speed
signal_length = 1e-3; % Signal length
Time = 30*signal_length; % PRI
prf = 1/Time;
snr = 1;
fc = 1e5; %  frequency
transmitter_power_per_pulse = 50; % Transmitted signal power
pulse_number = 10; % Number of pulses 
t = 0:dt:Time;
antenna_gain_p = 10*log10(antenna_gain); % Antenna gain in linear

signal = dsp.Chirp('SweepDirection','Bidirectional','TargetFrequency',fc,'InitialFrequency',0,'TargetTime',signal_length,'SweepTime',signal_length,'SamplesPerFrame',length(t),'SampleRate',Fs);
window = 1*(t>=0 & t<=signal_length);
wave = signal()'.*window;
power_t = sum((wave).^2)/(signal_length);
k_square = transmitter_power_per_pulse*signal_length/sum((wave).^2);
wave = wave*sqrt(k_square);
power_t_new = sum((wave).^2)/(signal_length);

free_spl = (4*pi*Range*fc/light_speed)^2; % Free space path loss
loss = antenna_gain_p^2*rcs/(free_spl*4*pi*Range^2); % Total loss from radar equation
rpower = transmitter_power_per_pulse*loss; % Received signal power
delay = -2*Range/light_speed; % Delay of received signal
window_received = 1*((t+delay)>=0 & (t+delay)<=signal_length); % Received signal width
signal_model_r = signal()'.*window_received;
receiver_value_square = rpower*signal_length/sum(signal_model_r.^2);
received_signal = signal_model_r*sqrt(receiver_value_square); % Received signal
receiver_power = sum(received_signal.^2)/length(t); % Received signal power
%received_signal_w_noise = awgn(received_signal,snr,10*log10(receiver_power));
received_signal_w_noise = received_signal + 2e-6*randn(1,length(t)); % Received signal with noise 2
power_of_noise = sum((received_signal_w_noise-received_signal).^2)/length(t); % Power of noise 1
snr_db_check = 10*log10(receiver_power/power_of_noise);

figure;
plot(t,wave);

figure;
plot(t,received_signal);

figure;
plot(t,received_signal_w_noise);

h = wave(end:-1:1).';
filt = conv(h,received_signal);
figure;
plot(t,filt(round(length(filt)/2):end));

filtered = conv(h,received_signal_w_noise);
figure;
plot(t,filtered(round(length(filtered)/2):end));

guard_cell_number = 1300;
reference_cell_number = 20;
cfar_vec = [ones(1,reference_cell_number),zeros(1,guard_cell_number),ones(1,reference_cell_number)];
cfar_vec = cfar_vec/sum(cfar_vec);
pfa = 1e-6;
N = 2*reference_cell_number;
alpha = N*(pfa^(-1/N)-1);
c_1 = alpha*conv(abs(filtered),cfar_vec,'same');
figure;
plot(c_1);
hold on;
plot(filtered);
legend("1:cfar","2:sinyal");


figure;
plot(10*log10(c_1));
hold on;
plot(10*log10(abs(filtered)));
legend("1:cfar","2:sinyal");



