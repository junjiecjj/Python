function varargout = FMCW_radar(d_m, v_m, azimuth)
%% Ali Karimzadeh Esfahani
c = physconst('LightSpeed');    % Speed of light in air (m/s)
Dres = 0.3;    % Distance Resulotion(m)
Dmax = 300;    % Maximum Distance(m)
Vmax = 200/3.6;     % Maximum Speed (m/s)
Vres = 1.56/3.6;     % Speed Resulotion(m/s)

target_D = d_m;

nb_sensors = 16;   % Number of Sensors
obs_period = 34.56e-3;    % observation period [1]p612
carrier_freq = 1e10;    % carrier frequency
% K_chirp = 3.7e12; 
lambda_carrier = c / carrier_freq;
d = lambda_carrier/2;

BW_chirp = c/(2*Dres); % bandwidth of LMCW signal used (in each chirp)
% Hz, bandwidth (eq. 1, "Radar Sensor Signal Acquisition and
% Multidimensional FFT Processing for Surveillance Applications in
% Transport Systems") confirmed for LMCW radar
% https://dsp.stackexchange.com/questions/50431/deriving-the-resolution-equation-of-an-fmcw-radar)

nb_chirps = 2 * Vmax / Vres; % number of chirps

T_chirp = obs_period/nb_chirps; % time length of each chirp

K_chirp = BW_chirp/T_chirp; % modulation index of chirps

f_beat_max = abs(2*Vmax/lambda_carrier) + K_chirp * 2 * Dmax/c; %  max beat frequency
% equation fbeat = -2*V/lambda_carrier + K_chirp * 2 * D/c resulting from
% linearly increasing inst. frequency, valid in steady state section when
% both emitter and receiver signals observed in beat signal
% (eq. 4, "Radar Sensor Signal Acquisition and Multidimensional FFT
% Processing for Surveillance Applications in Transport Systems")

sampling_rate = 2 * f_beat_max; %sampling_rate = 15e6;

chirp_nb_samples = T_chirp * sampling_rate; % length of each chirp in samples

Vres = lambda_carrier/(2*nb_chirps*T_chirp);
% (?) eq.3.3 "LMCW-radar. Signal processing and parameter estimation". Here
% will give same result as initial specification if lambda_carrier=d (with
% spatial_aliasing) but factor of 2 difference if deduced from data in p.3
% left column with T_chirp replaced by "nb_chirp_samples / fs", "Radar
% Image Reconstruction from Raw ADC Data using Parametric Variational
% Autoencoder with Domain Adaptation"

Vmax = Vres * (nb_chirps/2); 
% from previously used nb_chirps = 2* round(Vmax_kmh / Vres_kmh); % p.609
% 1st par 2nd column, "Radar Sensor Signal Acquisition and Multidimensional
% FFT Processing for Surveillance Applications in Transport Systems"

Dmax = Dres * (chirp_nb_samples/2); 
% similar form to above, deduced from p.3 left column,
%"Radar_Image_Reconstruction_from_Raw_ADC_Data_using_Parametric_Variational_Autoencoder_with_Domain_Adaptation"
% Dres depends only on BW and c, so it will be valid and Dmax can be
% computed from it.


theta_res_worst_case = lambda_carrier / (2*(nb_sensors-1)*d); 
% rad. (angle dependent, worst case of theta_res = lambda_carrier /
% (2*(nb_sensors-1)*d*cos(theta)), eq. 2, "Radar Sensor Signal Acquisition
% and Multidimensional FFT Processing for Surveillance Applications in
% Transport Systems")

theta_res_degrees_worst_case = theta_res_worst_case /pi*180; % degrees
target_cross_range_res_worst_case = theta_res_worst_case * target_D;
% m, eq. 3, "Radar Sensor Signal Acquisition and Multidimensional FFT
% Processing for Surveillance Applications in Transport Systems")

%% y_lk Calculation
for n = 1 : floor(T_chirp*sampling_rate)
    for l = 1 : nb_chirps
        for k = 1 : nb_sensors
            y_lk(n,l,k) = cos(2*pi*(2*(BW_chirp*d_m/T_chirp/c)*(n-1)/sampling_rate+2*(v_m/lambda_carrier)*(l-1)*T_chirp+(d*sin(azimuth)/lambda_carrier)*(k-1))); % ds?
        end
    end
end
nOutputs = nargout;
varargout = cell(1,nOutputs);

if nOutputs == 1
    varargout{1} = y_lk;
elseif  nOutputs == 7
    varargout{1} = y_lk;
    varargout{2} = Dres;
    varargout{3} = Dmax;
    varargout{4} = Vres;
    varargout{5} = Vmax;
    varargout{6} = theta_res_degrees_worst_case;
    varargout{7} = target_cross_range_res_worst_case;
end
end