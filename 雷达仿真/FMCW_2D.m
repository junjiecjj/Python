% https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247503888&idx=5&sn=91128cb13ac6630210d6b8b6499d4e64&chksm=c2ed302e9bac292547e68672237cf6c0cd225ce1341b7b3729cbe9a3b527092397fcee36e16f&mpshare=1&scene=1&srcid=0418eJ3yhGBfCih3fAoII6m8&sharer_shareinfo=8b359d0170425ccb2bde8c4af91d4c4b&sharer_shareinfo_first=8b359d0170425ccb2bde8c4af91d4c4b&exportkey=n_ChQIAhIQWRiY%2Fy5a7x2s3ez1NEpK9BKfAgIE97dBBAEAAAAAANFbCZdm9S0AAAAOpnltbLcz9gKNyK89dVj0UcD%2B%2BkUEbFZE1wQghzG8A%2BZdsriQPSbPcdbYuPLqEhN%2FArUqtVYA9UJt13wnxHKC5VHMjtIKs7cqkz%2B2qFltxVtNFZF9AUFS4GeXdkf%2FUeCuM2OHh9V7mT8YNqx4WRWYgldh0WwTqbyPsdH4uC8d9Y0lDGbW0KoEmO8vOYfVscVQvL0fR0vKSv4nsyqnvRtpAfOTCXJtRivCdwBs72laJHTwhqGxnaMQo6ojdDCh9RDbKFcag2It99L0qFuxmSFo%2FNOh1J1UL6m7eeAPz8Q%2BG6pOcZtnIdaxOcHKH6fHY30n1fGzd%2BA9YifyB%2B8kAQkcY858ih9sPFMF&acctmode=0&pass_ticket=gOtBYAlXM34c%2FAP6uuptKu57VPg8uNOrz0qOvCVuc9ipvqAOJQ6bUnW0DZyuQOnO&wx_header=0#rd
clc;
clear all;
close all;
%It is for FMCW Radars
%% Define Radar parameters
radar_parameters.max_range = 25; % Maximum Unambiguous Range in meters
radar_parameters.max_velocity =20; % Maximum Velocity in m/s
radar_parameters.L = 128; % Number of Chirps
radar_parameters.N = 256; % Number of Samples per Chirp
radar_parameters.B = 1500e6; % Sweep Bandwidth in Hz
radar_parameters.f_start = 78e9; % Carrier Frequency in Hz
radar_parameters.T= 50e-6; % Chirp Duration in seconds
radar_parameters.TRRI = 60e-6;% Ramp Repetion Interval
radar_parameters.f_s = 5e6; % ADC Sampling Rate in Hz
radar_parameters.T_s = 1/radar_parameters.f_s;
radar_parameters.S = radar_parameters.B/radar_parameters.T; % Ramp Slope in Hz/s
target_range=5;
target_velocity=10;
c = 3e8; % Speed of light in m/s
%% Call the function adcDataGenerate
Rx = adcDataGenerate(radar_parameters,target_range,target_velocity);
%% Noise Generation
signal_power = 1; % Signal power
SNR_dB = 20; % SNR in dB

% Convert SNR from dB to linear scale
SNR_linear = db2pow(SNR_dB);

% Calculate noise power
noise_power = signal_power / SNR_linear;

% Calculate standard deviation from noise power(mean=0)
sigma = sqrt(noise_power / 2);

% Generate real and imaginary parts
real_part = sigma * randn(radar_parameters.N, radar_parameters.L); 
imaginary_part = sigma * randn(radar_parameters.N, radar_parameters.L); 

% Combining real and imaginary parts
Noise=complex(real_part,imaginary_part);

X=Rx+Noise;
%% DFT Signal Processing
F_n = (1/sqrt(radar_parameters.N))*dftmtx(radar_parameters.N); %DFT matrix with N*N
F_l = (1/sqrt(radar_parameters.L))*dftmtx(radar_parameters.L); %DFT matrix with L*L
X_2d = (F_n)*X*(transpose(F_l)); %The expression for finding the 2D DFT 

% Range Resolution
del_R = (radar_parameters.T * c) / (2 * radar_parameters.B*radar_parameters.T_s*radar_parameters.N);

% Velocity Resolution
del_v = c / (2 * radar_parameters.f_start * radar_parameters.TRRI * radar_parameters.L);

%Range and velocity limits
R_axis = 0:del_R:((radar_parameters.N/2)-1)* del_R ;
v_axis = -del_v*(radar_parameters.L/2) :del_v:del_v*((radar_parameters.L/2) - 1);
[v_grid,R_grid] = meshgrid(v_axis,R_axis);

%Accessing only N/2 DFT coeffients
X_2d = X_2d(1:radar_parameters.N/2, :);

% Define the limits for column swapping
first_half = 1:radar_parameters.L/2;
second_half = (radar_parameters.L/2)+1:radar_parameters.L;

% Swap columns with defined limits
X_2d(:, [first_half, second_half]) = X_2d(:, [second_half, first_half]);

% Frequency grids for plotting
k_axis = 0:(radar_parameters.N/2)-1;
p_axis = -radar_parameters.L/2 :(radar_parameters.L/2) - 1;
[p_grid, k_grid] = meshgrid(p_axis, k_axis);

% Plot magnitude of 2D DFT
figure;
surf(p_grid, k_grid, abs(X_2d), 'EdgeColor', 'none');
xlabel('p axis');
ylabel('k axis');
zlabel('Magnitude');
title('3D Plot of 2D DFT Magnitude in frequency indices');

%% Plot magnitude of 2D DFT
figure;
surf(v_grid, R_grid, abs(X_2d), 'EdgeColor', 'none');
xlabel('Velocity (m/s)');
ylabel('Range (m)');
zlabel('Magnitude');
title('3D Plot of 2D DFT Magnitude in Range and Velocity');

R_max=(radar_parameters.f_s*c)/(2*radar_parameters.S);
disp(R_max);
v_max=c/(4*radar_parameters.T*radar_parameters.f_start);
disp(v_max);

% Function adcDataGenerate that simulates the data from the given structue radarparamets
function Rx = adcDataGenerate(radar_parameters, target_range, target_velocity)
    c = 3e8;                   %  Speed of light in m/s
    N = radar_parameters.N;
    L = radar_parameters.L; 
    T = radar_parameters.T;
    B = radar_parameters.B;
    max_range = radar_parameters.max_range;
    f_start = radar_parameters.f_start;
    TRRI = radar_parameters.TRRI;
    S = radar_parameters.S;
    f_s = radar_parameters.f_s;
    T_s = radar_parameters.T_s;              % Sampling Period
    R = target_range;
    v = target_velocity; 
    max_prop_time = 3 *max_range/c;        % Maximum delay
    for Chi_idx=1:L                        % Chirp index for L Chirps
        for Sam_idx=1:N                    % Sample index for N Samples
            t = max_prop_time +(Sam_idx-1)*T_s;
            phase_Tx(Sam_idx,Chi_idx) = 2*pi*f_start*t+pi*S*t^2;
            % Phase  of the received Signal
            tau = 2*(R+v*t+v*TRRI*(Chi_idx-1))/c;
            phase_Rx(Sam_idx,Chi_idx) = 2 * pi * f_start *(t-tau) + pi * S * (t-tau)^2;
            % Phase of the mixed Signal/IF signal 
            phase_IF(Sam_idx,Chi_idx)=phase_Tx(Sam_idx,Chi_idx)- phase_Rx(Sam_idx,Chi_idx); 
            % The Mixed Signal/ IF signal
            Rx(Sam_idx,Chi_idx)= complex(cos(phase_IF(Sam_idx,Chi_idx)),sin(phase_IF(Sam_idx,Chi_idx))); % Complex command
        end
    end 
end
