%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

%% Basic ISAC System Setting

% Note: To make the simulation as practical as possible, I used the mature MATLAB toolbox for communucations in simulation.
%       So, if you do not think my simulations are practical, please argue with the MATLAB toolbox development team!
%       Of course, the above suggestion is just a joke.

% Set random number generator for reproducibility
rng(68);        % In my culture, 6 means luck, 8 means richness. Is this a good reason to use 68 here?
                % If you mind, try 28 because I am 28 year old now! 
                % If you mind again, try 49 because my dear mom is 49 year old now!
                % If you still mind, time to say "Bad Bye"! Please close all my codes!

%% Basic System Setups
% If you want to know why I use the following settings for (fc,Pt,SNR), I suggest you ask [H. Zhang et. al. (2020; TCCN)]
fc = 5.9e9;                         % Carrier frequency (Hz)
Pt = 2.5;                           % Total Peak transmit power (W); Pt does not matter; only SNR matters.
lambda = freq2wavelen(fc);          % Wavelength (m)

N = 16;                             % Number of array elements; ask [F. Liu et. al. (2018; TSP)] for the reason
d = 0.5*lambda;                     % Array element spacing (m)

% Construct the transmit antenna array (ISAC Base Station), using isotropic antenna elements
element = phased.IsotropicAntennaElement('BackBaffled', true);
array = phased.ULA('Element', element, 'NumElements', N, 'ElementSpacing', d, 'ArrayAxis', 'y');
tx_pos = array.getElementPosition();

% Matrix of steering vectors corresponding to the angles in the grid ang
normalizedPos = tx_pos/lambda;
ang = linspace(-90, 90, 200);       % Grid of azimuth angles
A_steer_vec = steervec(normalizedPos, [ang; zeros(size(ang))]);

%% Objective of the Communication Component
% If you want to know why I use the following settings for (K,L,Q,S), I suggest you ask [F. Liu et. al. (2018; TSP)]
K = 4;                              % Number of communication users
L = 30;                             % Number of communication symbols

Q = 4;                              % Number of type of symbols (i.e., how many points in constellation)
data = randi([0 Q-1], K, L);        % Binary data
S = sqrt(Pt)*pskmod(data, Q, pi/Q);       % QPSK symbols

% Users' locations are random; I set the following myself without the agreement from the MATLAB toolbox; but I do not think this is a big issue.
rx_pos = [
       rand(1, K)     *1000
    (2*rand(1, K) - 1)*1000
                 zeros(1, K)        % No vertical component; only the 2-dimensional case
];
rx_azm = atan(rx_pos(2, :)./rx_pos(1, :))*180/pi;

% Downlink Multi-User Communication Channel: Create a scattering channel matrix assuming "numscat" independent scatterers
% Let's suppose both H0 and HBar are drawn from the ball "|H - Href| <= theta" for some norm "|.|" and radius "theta"
numscat = randi([60, 100]);                      % Number of scatters; I found no much influence on results
HRef = scatteringchanmtx(tx_pos, rx_pos, numscat).'; 

% Nominal Channel
epsilon = 0.05;        % The uncertainty range
% You only have one nominal channel when doing system analyses and design
HBar = HRef + ((2*rand(size(HRef)) - 1) + (2*rand(size(HRef)) - 1) * 1j)*epsilon;

% Zero-Forcing precoder is optimal for communication; optimal in terms of minimum MUI
Xc = HBar'*(HBar*HBar')^-1*S;
Bc = GetBeamPattern(Xc*Xc'/L, A_steer_vec);

% Signal-to-Noise Ratio
SNR = 10;   % 24 dB

% Channel noise level
N0 = Pt/(10^(SNR/10));

% Channel capacity: When zero-forcing precoding, i.e., X = H'*(H*H')^-1*S, used, the MUI energy equals zero; HX - S = S - S = 0.
C = log2(1 + 10^(SNR/10));

%% Objective of the Radar Component
% Three targets of interest (real azimuths)
tgtAz = [-45, 45];                  % Azimuths of the targets of interest
tgtRng = [1.51e3, 1.39e3];          % Ranges of the targets of interest

beamwidth = 10;                     % Desired beamwidth

% Desired beam pattern
idx = false(size(ang));
for i = 1:numel(tgtAz)
    idx = idx | ang >= tgtAz(i)-beamwidth/2 & ang <= tgtAz(i)+beamwidth/2;
end

Bdes = zeros(size(ang)) + 0.01;     % NB: 10*log(0) = -infty; so use a small value to replace it
Bdes(idx) = 1;

% MIMO Radar Waveform Synthesis: use desired beampattern "Bdes" to generate a practical beampattern "B" (Using minimum-mean-square-error approach)
R = GetR(normalizedPos, Bdes, ang, Pt, N);
B = GetBeamPattern(R, A_steer_vec);

% From Covariance Matrix to an Ideal Radar Waveform Xs (Using the cyclic algorithm with constrained peak-to-average power ratio)
Xs = GetXs(R, L);
Rs = Xs*Xs'/L;
Bs = GetBeamPattern(Rs, A_steer_vec);

%% Performance Comparison for Separate Waveform Design
disp('-----------------------------');
% True Channel for performance testing (NB: use nominal channel for design, while use true channel for test)
H0 = HRef + ((2*rand(size(HRef)) - 1) + (2*rand(size(HRef)) - 1) * 1j)*epsilon;
% Sensing
newTgtAz = -45;
disp(['Detection Probability of Target at ' num2str(newTgtAz) ' degree:']);
[~, idx] = min(abs(ang - newTgtAz));
disp(['     using Perfect Sensing       Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), Xs, N0, 1e-3))]);
disp(['     using Perfect Communication Waveform : ' num2str(DetectProb(A_steer_vec(:, idx), Xc, N0, 1e-3))]);
% Communication
disp('Average Achievable Sum-Rate');
disp(['     using Perfect Sensing       Waveform : ' num2str(GetAvrgSumRate(H0, Xs, S, N0))]);
disp(['     using Perfect Communication Waveform : ' num2str(GetAvrgSumRate(H0, Xc, S, N0))]);
disp('-----------------------------');

%% Plots
% Plot the simulation scenario
figure;
plot(tx_pos(1, :), tx_pos(2, :), 'm>', 'markersize', 8, 'linewidth', 2);
hold on;
plot(rx_pos(1, :), rx_pos(2, :), 'b*', 'markersize', 8, 'linewidth', 2);
tgt_pos = [tgtRng .* cos(tgtAz.*pi/180); tgtRng .* sin(tgtAz.*pi/180)];
plot(tgt_pos(1, :), tgt_pos(2, :), 'rs', 'markersize', 8, 'linewidth', 2);
axis([0 2 -3 3]*1e3);
set(gca, 'fontsize', 16);
xlabel('$x$ coordinate (meter)', 'interpreter', 'latex');
ylabel('$y$ coordinate (meter)', 'interpreter', 'latex');
box on;

% Plot beam patterns for perfect sensing only
figure;
hold on
plot(ang, pow2db(Bdes/max(Bdes)), 'b--', 'LineWidth', 2);
plot(ang, pow2db(Bs/max(Bs)), 'r', 'LineWidth', 2);
grid on
xlabel('Azimuth (deg)')
ylabel('(dB)')
legend('Theoretically Perfect Sensing Beampattern', 'Practically Perfect Sensing Beampattern')
set(gca, 'fontsize', 16)
ylim([-30 1])
box on;

% Plot beam patterns for perfect communication only
figure;
hold on
plot(ang, pow2db(Bc/max(Bc)), 'm', 'LineWidth', 2); 
plot(ang, pow2db(Bdes/max(Bdes)), 'b--', 'LineWidth', 2);
xline(rx_azm, 'k-.', 'linewidth', 2);
grid on
xlabel('Azimuth (deg)')
ylabel('(dB)')
legend('Perfect Communication (Zero Forcing)', 'Theoretically Perfect Sensing Beampattern', 'User Azimuths')
ylim([-30 1])
set(gca, 'fontsize', 16);
box on;

