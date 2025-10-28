%% 到达方向估计：波束扫描、MVDR与MUSIC方法
% 本示例展示了如何应用波束扫描、最小方差无失真响应（MVDR）和多信号分类（MUSIC）进行到达方向（DOA）的估计。
% 波束扫描是一种形成常规波束并在感兴趣的区域内扫描以获取空间谱的方法。
% MVDR与波束扫描类似，但使用的是MVDR波束，能提供更优的性能。
% 多信号分类（MUSIC）是一种子空间技术，能提供高精度的DOA估计。
% 对于这三种方法，输出的空间谱的峰值都对应着接收到的信号的DOA。
% 在这个例子中，我们将展示如何使用波束扫描、MVDR和MUSIC来估计均匀线性阵列（ULA）的端射角，以及均匀矩形阵列（URA）的方位角和仰角。

clc; clear; close all;

%% 建立均匀线性阵列（ULA）接收信号模型
% 首先，模拟一个包含10个等向性天线的均匀线性阵列（ULA），天线之间的间距为0.5米。

ula = phased.ULA('NumElements', 10, 'ElementSpacing', 0.5);

%% 参数配置
ang1 = [40; 0]; % 方位角40度、仰角0度
ang2 = [-20; 0]; % 方位角-20度、仰角0度
angs = [ang1 ang2];

c = physconst('LightSpeed'); % 光速
fc = 300e6; % 载波频率
lambda = c / fc; %载波波长
pos = getElementPosition(ula) / lambda;
Nsamp = 1000; % 采样点数

%%
% 同时，假设每个天线的热噪声功率为0.01w。
nPower = 0.01;

%% 生成多信号源均匀线阵的接受信号

rs = rng(2007);

% pos:阵元位置，单位为信号波长
% Nsamp:采样点数
% angs:接收信号的方位角、仰角
% nPower:噪声功率
signal = sensorsig(pos, Nsamp, angs, nPower);

%% 使用ULA进行波束扫描DOA估计
% 我们希望通过接收到的信号来估计这两个DOA（方向-of-arrival）。由于信号是由对称于其轴线的ULA（均匀线阵）接收的，我们无法同时得到方位角和仰角。相反，我们可以估计宽边角，这是从ULA的宽边测量的角度。这些角度之间的关系如下图所示：
broadsideAngle = az2broadside(angs(1, :), angs(2, :))

%%
% We can see that the two broadside angles are the same as the azimuth
% angles. In general, when the elevation angle is zero and the azimuth
% angle is within [-90 90], the broadside angle is the same as the azimuth
% angle. In the following we only perform the conversion when they are not
% equal.

%%
% 波束扫描算法通过一个预定义的扫描区域对常规波束进行扫描。在这里，我们将扫描区域设置为[-90 90]，以覆盖全部180度。

spatialspectrum = phased.BeamscanEstimator('SensorArray', ula, ...
    'OperatingFrequency', fc, 'ScanAngles', -90:90);

%%
% By default, the beamscan estimator only produces a spatial spectrum
% across the scan region. Set the DOAOutputPort property to true to
% obtain DOA estimates. Set the NumSignals property to 2 to find
% the locations of the top two peaks.

spatialspectrum.DOAOutputPort = true;
spatialspectrum.NumSignals = 2;

%%
% We now obtain the spatial spectrum and the DOAs. The estimated DOAs
% show the correct values, which are 40&deg; and -20&deg;.

[~, ang] = spatialspectrum(signal)

%%
% Plot the spatial spectrum of the beamscan output.

plotSpectrum(spatialspectrum);

%% Improving Resolution Using MVDR and MUSIC Estimators
% The conventional beam cannot resolve two closely-spaced signals.
% When two signals arrive from directions separated by less than the
% beamwidth, beamscan will fail to estimate the directions of the
% signals. To illustrate this limitation, we simulate two received signals
% from 30&deg; and 40&deg; in azimuth.

ang1 = [30; 0]; ang2 = [40; 0];
signal = sensorsig(pos, Nsamp, [ang1 ang2], nPower);

[~, ang] = spatialspectrum(signal)

%%
% The results differ from the true azimuth angles. Let's take a look
% at the output spectrum.

plotSpectrum(spatialspectrum);

%%
% The output spatial spectrum has only one dominant peak. Therefore, it
% cannot resolve these two closely-spaced signals. When we try to estimate
% the DOA from the peaks of the beamscan output, we get incorrect
% estimates. The beamscan object returns two maximum peaks as the estimated
% DOAs no matter how different the peaks are. In this case, the beamscan
% returns the small peak at 71&deg; as the second estimate.

%%
% To resolve closely-spaced signals, we can use the minimum variance
% distortionless response (MVDR) algorithm or the multiple signal
% classification (MUSIC) algorithm. First, we will examine the MVDR
% estimator, which scans an MVDR beam over the specified region. Because an
% MVDR beam has a smaller beamwidth, it has higher resolution.

mvdrspatialspect = phased.MVDREstimator('SensorArray', ula, ...
    'OperatingFrequency', fc, 'ScanAngles', -90:90, ...
    'DOAOutputPort', true, 'NumSignals', 2);
[~, ang] = mvdrspatialspect(signal)
plotSpectrum(mvdrspatialspect);

%%
% The MVDR algorithm correctly estimates the DOAs that are unresolvable by
% beamscan. The improved resolution comes with a price. The MVDR is more
% sensitive to sensor position errors. In circumstances where sensor
% positions are inaccurate, MVDR could produce a worse spatial spectrum
% than beamscan. Moreover, if we further reduce the difference of two
% signal directions to a level that is smaller than the beamwidth of an
% MVDR beam, the MVDR estimator will also fail.

%%
% The MUSIC algorithm can also be used to resolve these closely-spaced
% signals. Estimate the directions of arrival of the two sources and
% compare the spatial spectrum of MVDR to the spatial spectrum of MUSIC.

musicspatialspect = phased.MUSICEstimator('SensorArray', ula, ...
    'OperatingFrequency', fc, 'ScanAngles', -90:90, ...
    'DOAOutputPort', true, 'NumSignalsSource', 'Property', 'NumSignals', 2);
[~, ang] = musicspatialspect(signal)
ymvdr = mvdrspatialspect(signal);
ymusic = musicspatialspect(signal);
helperPlotDOASpectra(mvdrspatialspect.ScanAngles, ...
    musicspatialspect.ScanAngles, ymvdr, ymusic, 'ULA')

%%
% The directions of arrival using MUSIC are correct, and MUSIC provides
% even better spatial resolution than MVDR. MUSIC, like MVDR, is sensitive
% to sensor position errors. In addition, the number of sources must be
% known or accurately estimated. When the number of sources specified is
% incorrect, MVDR and Beamscan may simply return insignificant peaks from
% the correct spatial spectrum. In contrast, the MUSIC spatial spectrum
% itself may be inaccurate when the number of sources is not specified
% correctly. In addition, the amplitudes of MUSIC spectral peaks
% cannot be interpreted as the power of the sources.
%%
% For a ULA, additional high resolution algorithms can further exploit the
% special geometry of the ULA. See
% <docid:phased_ug.example-ex70791999 High Resolution Direction of Arrival Estimation>.

%% Converting Broadside Angles to Azimuth
% Although we can only estimate broadside angles using a ULA, we can
% convert the estimated broadside angles to azimuth angles if we know their
% incoming elevations. We now model two signals coming from 35&deg; in
% elevation and estimate their corresponding broadside angles.

ang1 = [40; 35]; ang2 = [15; 35];

signal = sensorsig(pos, Nsamp, [ang1 ang2], nPower);
[~, ang] = mvdrspatialspect(signal)

%%
% The resulting broadside angles are different from either the azimuth or
% elevation angles. We can convert the broadside angles to the azimuth
% angles if we know the elevation.

ang = broadside2az(ang, 35)

%% Beamscan DOA Estimation with a URA
% Next, we illustrate DOA estimation using a 10-by-5 uniform rectangular
% array (URA). A URA can estimate both azimuth and elevation angles. The
% element spacing is 0.3 meters between each row, and 0.5 meters between
% each column.

ura = phased.URA('Size', [10 5], 'ElementSpacing', [0.3 0.5]);

%%
% Assume that two signals impinge on the URA. The first signal arrives from
% 40&deg; in azimuth and 45&deg; in elevation, while the second
% signal arrives from -20&deg; in azimuth and 20&deg; in elevation.

ang1 = [40; 45]; % First signal
ang2 = [-20; 20]; % Second signal

signal = sensorsig(getElementPosition(ura) / lambda, Nsamp, ...
    [ang1 ang2], nPower);
rng(rs); % Restore random number generator

%%
% Create a 2-D beamscan estimator object from the URA. This object uses the
% same algorithm as the 1-D case except that it scans over both azimuth and
% elevation instead of broadside angles.
%
% The scanning region is specified by the property 'AzimuthScanAngles' and
% 'ElevationScanAngles'. To reduce computational complexity, we assume some
% a priori knowledge about the incoming signal directions. We restrict the
% azimuth scan region to [-45 45] and the elevation scan region to [10 60].

azelspectrum = phased.BeamscanEstimator2D('SensorArray', ura, ...
    'OperatingFrequency', fc, ...
    'AzimuthScanAngles', -45:45, 'ElevationScanAngles', 10:60, ...
    'DOAOutputPort', true, 'NumSignals', 2);

%%
% The DOA output is a 2-by-N matrix where N is the number of signal
% directions. The first row contains azimuth angles while the second
% row contains elevation angles.

[~, ang] = azelspectrum(signal)

%%
% Plot a 3-D spectrum in azimuth and elevation.

plotSpectrum(azelspectrum);

%% MVDR DOA Estimation with a URA
% Similar to the ULA case, we use a 2-D version of the MVDR algorithm.
% Since our knowledge of the sensor positions is perfect, we expect the
% MVDR spectrum to have a better resolution than beamscan.

mvdrazelspectrum = phased.MVDREstimator2D('SensorArray', ura, ...
    'OperatingFrequency', fc, ...
    'AzimuthScanAngles', -45:45, 'ElevationScanAngles', 10:60, ...
    'DOAOutputPort', true, 'NumSignals', 2);
[~, ang] = mvdrazelspectrum(signal)
plotSpectrum(mvdrazelspectrum);

%% MUSIC DOA Estimation with a URA
% We can also use the MUSIC algorithm to estimate the directions of arrival
% of the two sources.
musicazelspectrum = phased.MUSICEstimator2D('SensorArray', ura, ...
    'OperatingFrequency', fc, ...
    'AzimuthScanAngles', -45:45, 'ElevationScanAngles', 10:60, ...
    'DOAOutputPort', true, 'NumSignalsSource', 'Property', 'NumSignals', 2);
[~, ang] = musicazelspectrum(signal)
plotSpectrum(musicazelspectrum);

%%
% To compare MVDR and MUSIC estimators, let's consider sources located even
% closer together. Using MVDR and MUSIC, compute the spatial spectrum
% of two sources located at 10&deg; in azimuth and separated by 3&deg;
% in elevation.
ang1 = [10; 20]; % First signal
ang2 = [10; 23]; % Second signal

signal = sensorsig(getElementPosition(ura) / lambda, Nsamp, ...
    [ang1 ang2], nPower);
[~, angmvdr] = mvdrazelspectrum(signal)
[~, angmusic] = musicazelspectrum(signal)

%%
% In this case, only MUSIC correctly estimates to directions of arrival for
% the two sources. To see why, plot an elevation cut of each spatial
% spectrum at 10&deg; azimuth.
ymvdr = mvdrazelspectrum(signal);
ymusic = musicazelspectrum(signal);
helperPlotDOASpectra(mvdrazelspectrum.ElevationScanAngles, ...
    musicazelspectrum.ElevationScanAngles, ymvdr(:, 56), ymusic(:, 56), 'URA')

%% Summary
% 在此示例中，我们展示了如何将波束扫描、MVDR以及MUSIC技术应用于DOA估计问题。我们使用这两种技术为ULA接收到的信号估计了宽边角。在没有传感器位置误差的情况下，MVDR算法比波束扫描具有更高的分辨率。MUSIC的分辨率甚至优于MVDR，但必须知道信号源的数量。此外，我们还说明了方位角与宽边角之间的转换方法。接下来，我们使用波束扫描、MVDR以及MUSIC技术，通过URA估计方位角和仰角。在所有这些情况下，我们都绘制了输出的空间谱，并再次发现MUSIC具有最佳的空间分辨率。波束扫描、MVDR和MUSIC技术可应用于任何类型的阵列，但对于ULA和URA，还有额外的高分辨率技术可以进一步利用阵列的几何结构。
