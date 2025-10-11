clear all;
clc;
% Victim radar parameters
fc = 76.5e9; % Start frequency
bw_sv = 500e6; % Bandwidth
tm_sv = 50e-6; % Sweep time
fs_sv = bw_sv; % Sampling frequency
N_sv = 25000; % Number of samples in each chirp
Nd_sv = 135; % Number of chirps
% Initialize variables for later use
N=N_sv;
Nd=Nd_sv;
fs=fs_sv;
c=3e8;
slope=bw_sv/tm_sv; % Frequency slope
lambda = c/fc; % Wavelength (m)
tm=tm_sv;
% Interferer radar parameters
fc_int = 76.45e9; % Start frequency
bw_int = 900e6; % Bandwidth
tm_int = 25e-6; % Sweep time
fs_int = bw_int; % Sampling frequency (Hz)
% Time vectors
t_sv = 0:1/fs_sv:(tm_sv*Nd_sv)-1/fs_sv; % Victim radar
% Transmitted signal (Victim radar)
% --- Target Parameters ---
t1_vel = 10; % Velocity of target 1 (m/s)
t2_vel = -5; % Velocity of target 2 (m/s)
t1_pos = [40, 3.6]; % Initial position of target 1 (m)
t2_pos = [20, 7.2]; % Initial position of target 2 (m)
victim_pos = [0, 0]; % Position of victim radar (m)
% Calculate initial distances from radar to targets
r1 = pdist([victim_pos; t1_pos], 'euclidean');
r2 = pdist([victim_pos;t2_pos ], 'euclidean');
range1 = r1 + t1_vel * t_sv; %change of range with respect to time
delta_t1 = 2 * range1 / 3e8; %change in delay time
t_chirp = 0:1/fs_sv:(tm_sv)-1/fs_sv; %time vectors for single chirp
t_int1=0:1/fs_int:tm_int-1/fs_int;% time vectors for interference radar
t_int= repmat(t_int1,[1,150]);
% --- Power and Amplitude Calculations ---
L = 4; % Length of the longest side of target (m)
L_int = 4.5; % Length of the longest side of interferer (m)
EIRP_dBW = -5; % Effective Isotropic Radiated Power (dBW)
EIRP_dBW_int = -5; % EIRP for interferer (dBW)
G_dBi = 10; % Antenna gain (dBi)
lambda_val = 0.0039; % Wavelength (m)
% Calculate Radar Cross Section (RCS)
rcs = 4 * pi * (L^4) / (3 * (lambda^2)); % RCS for target (m^2)
rcs_int = 4 * pi * (L_int^4) / (3 * (lambda^2)); % RCS for interferer (m^2)
% Convert EIRP and Gr from dB to linear scale
EIRP_linear = 10^(EIRP_dBW/10); % in Watts
EIRP_linear_int = 10^(EIRP_dBW_int/10); % in Watts
G_r_linear = 10^(G_dBi/10); % Linear scale
% Radar equation to compute Pr
Pr = (EIRP_linear * G_r_linear * lambda_val^2 * rcs) / ((4*pi)^3 * r1^4);
A = sqrt(2*Pr); % Amplitude for victim radar
% Radar equation to compute Pr of interference
Pr_int = (EIRP_linear * G_r_linear * lambda_val^2*rcs_int ) / ((4*pi)^2 * r2^2);
A_int = sqrt (2*Pr_int); % Amplitude for recieved radar
% --- Signal Generation for Each Chirp ---
for chirp_idx = 1:Nd_sv-1
 % Update the range for target 1 for each chirp
 r1(chirp_idx+1) = r1(chirp_idx) + t1_vel * tm_sv;
end
for chirp_idx = 1:Nd_sv
 % Calculate the time offset for the current chirp
 time_offset = (chirp_idx - 1) * tm_sv;
 
 % Calculate the time delay for target 1
 delta_t1 = 2 * r1(chirp_idx) / c;
 % Generate the transmitted signal for one chirp
 st_chirp = exp(1j * 2 * pi * (fc * t_chirp + (bw_sv / tm_sv) * (t_chirp.^2)/2));
 % Generate the received signal for one chirp considering target 1
 sr_chirp = A * exp(1j * 2 * pi * (fc * (t_chirp - delta_t1) + (bw_sv / tm_sv)*(t_chirp - delta_t1).^2 / 2 - 2 * fc * t1_vel / c));
 % Add noise to the received signal
 noise_level = A * randn(size(sr_chirp));
 sr_chirp1 = sr_chirp + noise_level;
 % Store the transmitted and received signals for this chirp
 st_all_chirps(:, chirp_idx) = st_chirp;
 sr_all_chirps(:, chirp_idx) = sr_chirp1;
end
% --- Generate Interference Signal ---
slope_int = bw_int / tm_int; % Frequency slope for interferer
% Generate the interference signal
sr_interference = A_int * exp(1j * 2 * pi * (fc_int * t_int + (slope_int / 2) * (t_int.^2) - 2 * fc_int * t2_vel / c));
% Reshape the interference signal to match dimensions
sr_int = reshape(sr_interference, [N, Nd]);
% Add the interference to the received signal
sr = sr_all_chirps + sr_int;
% --- Plotting and Visualization ---
%plot of transmitted signal
figure;
subplot(3,1,1);
plot(t_chirp, real(st_all_chirps(:, 1)));
axis([0 30e-6 -1.5 1.5]);
title('Transmitted Signal (Chirp 1)');
xlabel('Time (s)');
ylabel('Amplitude(m)');
%plot of recieved signal 
subplot(3,1,2);
plot(t_chirp, real(sr_all_chirps(:, 1)));
axis([0e-6 30e-6 -0.2e-2 0.2e-2]);
title('Received Signal (Chirp 1, Target 1)');
xlabel('Time (s)');
ylabel('Amplitude(m)');
%plot of recieved signal with interference
subplot(3,1,3);
plot(t_chirp, real(sr(:, 1)));
axis([0e-6 30e-6 -5e-1 5e-1]);
xlabel('Time (s)');
ylabel('Amplitude(m)');
title('Received Signal with interference (Chirp 1, Target 1)');
grid on;
%%%spectra of transmitted and recieved signal
% Transmitted Signal Spectrum
transmitted_spectrum = abs(fft(st_all_chirps(:,1)));
figure;
plot(20*log10(transmitted_spectrum));
title('Transmitted Signal Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
received_spectrum = abs(fft(sr_all_chirps(:,1)));
figure;
plot(20*log10(received_spectrum));
title('Received Signal Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
%%%5spectrogram plots
figure; received_signal = st_all_chirps(:); % Transmitted signal
spectrogram(received_signal, 'yaxis');
title('Spectrogram of transmitted Signal');
figure; received_signal = sr_all_chirps(:); % Recieved signal
spectrogram(received_signal, 'yaxis');
title('Spectrogram of Received Signal');
figure; received_signal = sr(:); % Recieved signal with interference
spectrogram(received_signal, 'yaxis');
title('Spectrogram of Received Signal with interference');
f_lpf=3.5e6;
order = 6;
ripple =0.5;
lpf_cutoff = f_lpf / (fs_sv/2);
%%mixer output
mix=st_all_chirps.*conj(sr_all_chirps);
mix_int1=st_all_chirps.*conj(sr);
%filtering
 [b_cheby, a_cheby] = cheby1(order, ripple, lpf_cutoff, 'low');
 mix1 = filter(b_cheby, a_cheby, mix);
 mix_int = filter(b_cheby, a_cheby, mix_int1); 
%amplification
mix2 = 3*mix1;
mix_int = 3*mix_int;
% --- Low-pass Filter Parameters ---
f_lpf = 3.5e6; % Low-pass filter cutoff frequency (Hz)
order = 6; % Filter order
ripple = 0.5; % Passband ripple (dB)
lpf_cutoff = f_lpf / (fs_sv / 2); % Normalized cutoff frequency
% --- Mixer Output ---
% Multiply the transmitted and received signals to get the mixed signal
mix = st_all_chirps .* conj(sr_all_chirps);
% Multiply the transmitted and received signals with interference to get the mixed signal with interference
mix_int1 = st_all_chirps .* conj(sr);
% --- Filtering ---
% Design a Chebyshev Type I low-pass filter
[b_cheby, a_cheby] = cheby1(order, ripple, lpf_cutoff, 'low');
% Apply the filter to the mixed signal
mix1 = filter(b_cheby, a_cheby, mix);
% Apply the filter to the mixed signal with interference
mix_int = filter(b_cheby, a_cheby, mix_int1);
% --- Amplification ---
% Amplify the filtered mixed signal by a factor of 3
mix2 = 3 * mix1;
% Amplify the filtered mixed signal with interference by a factor of 3
mix_int = 3 * mix_int;
% plot of IF signal without and with interference before filtering
figure
subplot(211); 
plot(t_sv,real(mix(:)));
axis([0e-6 30e-6 -5e-3 5e-3]);
xlabel('Time in s');
ylabel('Amplitude in m');
title('IF signal');
subplot(212);plot(t_chirp,real(mix_int1(:,1)));
axis([0e-6 30e-6 -4e-1 4e-1]);
xlabel('Time in s');
ylabel('Amplitude in m');
title('IF signal with interference');
% plot of IF signal without and with interference after filtering
figure
subplot(211); plot(t_sv,real(mix2(:)));
axis([0e-6 30e-6 -6e-3 6e-3]);
%axis([17e-6 25e-6 -10e-3 10e-3]);
xlabel('Time in s');
ylabel('Amplitudein m');
title('Amplified IF signal after filtering');
subplot(212);plot(t_sv,real(mix_int(:)));
axis([0e-6 30e-6 -14e-1 14e-1]);
xlabel('Time in s');
ylabel('Amplitude im ,');
title('Amplified IF signal with interference after filtering');
% --- Range Vector Calculation ---
% Calculate the frequency bins for FFT
fb = fs * (0:(N / 2) - 1) / N;
% Calculate the corresponding range values
r = c * fb / (2 * slope);
% --- Velocity Vector Calculation ---
% Calculate the maximum detectable velocity
v_max = lambda / (4 * tm);
% Calculate the velocity resolution
delta_v = lambda / (2 * Nd * tm);
% Generate the velocity vector
velocity_vec = -v_max:delta_v:v_max - delta_v;
% --- FFT for Range Detection ---
% Perform FFT on the mixed signal after amplification
sig_fft = abs((fft(mix2, N, 1)) ./ N);
% Take the first half of the FFT output
sig_fft1 = sig_fft(1:(N / 2), :);
% --- 1D FFT Plot (Without Interference) ---
% Plot the range information obtained from FFT
figure('Name', 'Range from First FFT');
plot(r, 20 .* log10(abs(sig_fft1(:, 1))), "LineWidth", 2);
grid on;
axis([0 200 -120 -40]);
xlabel('Range in m');
ylabel('FFT Output in dB');
title('1D FFT Without Interference');
% --- 2D FFT (Range-Doppler Map) ---
% Perform 2D FFT on the mixed signal
y = fft2(mix2, N, Nd);
y = y ./ N;
y = abs(y);
y = y(1:(N / 2), :);
y = ifftshift(y, 2);
% --- 2D FFT Plot (Without Interference) ---
figure;
s = surf(velocity_vec, r, 20 .* log10(abs(y)));
s.EdgeColor = 'none';
ylim([0 100]);
zlim([-90 0]);
caxis([-70 -5]);
xlabel('Velocity in m/s');
ylabel('Range in m');
zlabel('Magnitude FFT Output in dB');
title('2D FFT Without Interference');
% --- FFT for Range Detection with Interference ---
sig_fft_int = abs((fft(mix_int1, N, 1)) ./ N);
sig_fft1_int = sig_fft_int(1:(N / 2), :);
% --- 1D FFT Plot (With Interference) ---
figure('Name', 'Range from First FFT');
plot(r, 20 .* log10(abs(sig_fft1_int(:, 1))), "LineWidth", 2);
grid on;
axis([0 200 -100 -10]);
ylim([-44 -41]);
xlabel('Range in m');
ylabel('FFT Output in dB');
title('1D FFT With Interference');
% --- 2D FFT (Range-Doppler Map with Interference) ---
y_int = fft2(mix_int1, N, Nd);
y_int = abs(y_int ./ N);
y_int = y_int(1:(N / 2), :);
y_int = ifftshift(y_int, 2);
% --- 2D FFT Plot (With Interference) ---
figure;
m = surf(velocity_vec, r, 20 .* log10(abs(y_int)));
m.EdgeColor = 'none';
ylim([0 100]);
zlim([-110 0]);
caxis([-70 0]);
xlabel('Velocity in m/s');
ylabel('Range in m');
zlabel('Magnitude FFT Output in dB');
title('2D FFT With Interference');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Filter types and order evaluation%%%%%%%%%%%%%
st=st_all_chirps(:);
sr=sr_all_chirps(:);
sr_int1=reshape(sr_interference,[N,Nd]);
sr_int= sr_all_chirps+sr_int1;
%Mixer output
mix_int1=st_all_chirps.*conj(sr_int);
t=t_sv;
% LPF cutoff frequency
f_lpf = 3.5e6;
%lpf_cutoff = f_lpf / (fs/2); % Normalized cutoff frequency
lpf_cutoff = f_lpf / (fs/2);
% FIR Filter Design
orders = [4, 6, 8, 10];
figure;
c=1;
for order = orders
 b_fir = fir1(order, lpf_cutoff, 'low');
 filtered_output_fir = filter(b_fir, 1, mix_int1(:));
 subplot(4,1,c);
 c=c+1;
 plot(t, real(filtered_output_fir));
 axis([2.5e-4 7e-4 -5e-1 5e-1]);
 xlabel('Time');
 ylabel('Amplitude');
 title(['FIR Filtered Mixer Output (Order ' num2str(order) ')']);
 
end
% Butterworth Filter Design
figure;
c=1;
for order = orders
 
 [b_butter, a_butter] = butter(order, lpf_cutoff, 'low');
 filtered_output_butter = filter(b_butter, a_butter, mix_int1(:));
 subplot(4,1,c);
 plot(t, real(filtered_output_butter));
axis([4.65e-4 4.85e-4 -5e-1 5e-1]);
 xlabel('Time');
 ylabel('Amplitude');
 title(['Butterworth Filtered Mixer Output (Order ' num2str(order) ')']);
 c=c+1;
end
% Chebyshev Type I Filter Design
figure;
c=1;
ripple = 0.5; % Ripple in the passband in dB. Adjust as needed.
for order = orders
 [b_cheby, a_cheby] = cheby1(order, ripple, lpf_cutoff, 'low');
 filtered_output_cheby = filter(b_cheby, a_cheby, mix_int1(:));
 subplot(4,1,c);
 c=c+1;
 plot(t, real(filtered_output_cheby));
axis([4.65e-4 4.85e-4 -5e-1 5e-1]);
 xlabel('Time');
 ylabel('Amplitude');
 title(['Chebyshev Type I Filtered Mixer Output (Order ' num2str(order) ')']);
end