%% Plot MUSIC Spectrum of Two Signals Arriving at ULA
% Estimate the DOAs of two signals received by a standard 10-element ULA having 
% an element spacing of 1 meter. Then plot the MUSIC spectrum.
% 
% Create the ULA array. The antenna operating frequency is 150 MHz.

fc = 150.0e6;
array = phased.ULA('NumElements',10,'ElementSpacing',1.0);
%% 
% Create the arriving signals at the ULA. The true direction of arrival of the 
% first signal is 10° in azimuth and 20° in elevation. The direction of the second 
% signal is 60° in azimuth and -5° in elevation.

fs = 8000.0;
t = (0:1/fs:1).';
sig1 = cos(2*pi*t*300.0);
sig2 = cos(2*pi*t*400.0);
sig = collectPlaneWave(array,[sig1 sig2],[10 20; 60 -5]',fc);
noise = 0.1*(randn(size(sig)) + 1i*randn(size(sig)));
axis_angle = (-90:1:90);
signal = sig + noise;
%% 
% Estimate the DOAs. 

estimator = phased.MUSICEstimator('SensorArray',array,'ScanAngles', axis_angle,...
    'OperatingFrequency',fc,...
    'DOAOutputPort',true,'NumSignalsSource','Property',...
    'NumSignals',2);
[y,doas] = estimator(signal);
doas = broadside2az(sort(doas),[20 -5])

%%
% 获取阵元坐标
ula_pos = getElementPosition(array);
% 计算到达矢量
steering_vectors = steering_vector(ula_pos, [axis_angle; zeros(size(axis_angle))], physconst('LightSpeed')/fc);
% 去直流 转置
x = transpose(signal-mean(signal,1)) ; 
% 自协方差矩阵
Cxx = x * x';
[V, D] = eig(Cxx);
[D, I] = sort(diag(D), 'descend');
V = V(:, I);
Vn = V(:,3:end);
my_music_spec = 1 ./ (sum(abs((steering_vectors' * Vn)) .^ 2, 2));
my_music_spec = sqrt(my_music_spec);


%% 
% Plot the MUSIC spectrum.
figure("Name", "MUSIC 算法测角")
hold on;
helperPlotSpec(axis_angle, y, 'MUSIC 谱 (MATLAB函数)');
helperPlotSpec(axis_angle, my_music_spec, 'MUSIC 谱 (自己实现)', 'r*');
hold off;

%% 
% _Copyright 2012 The MathWorks, Inc._