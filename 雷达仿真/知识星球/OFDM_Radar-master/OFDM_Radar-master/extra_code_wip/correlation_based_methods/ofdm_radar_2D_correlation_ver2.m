%% Computing and plotting the Ambiguity function for the OFDM waveform:
% Based on the ideas discussed in: "Comparison of Correlation-based OFDM Radar Receivers"
% Most variable names are based on those used in the above paper.
% Trying to compute equation 4 of the paper.
K = 32; % The number of subcarriers being used
M = 5; % The frame size
modOrder = 16; % Order of modulation
d = (0:modOrder-1)';

% General constants:
c = 3e8; % The speed of light

% OFDM system parameters:
To = 4e-6; % The OFDM symbol period
fC = 5.5e9; % The frequency of the centre subcarrier
subcarrierSpacing = 312.5e3; % The subcarrier spacing

modulationType = "QPSK";

switch(modulationType)
    case "QAM"
        signalSet = qammod(d, modOrder); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)     
    case "QPSK"
        signalSet = pskmod(d, modOrder);
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % Exit the program early
end

Ftx = zeros(K/2, M); % Half of the empty transmit frame

for row = 2:K/2 % We start at row 2, as row 1 is empty for the DC subcarrier...
    Ftx(row, :) = randsample(signalSet, M, true);
end

% The full transmit frame. Here we are imposing Hermitian symmetry, so that
% the output from the IFFT block is real-valued... We add a middle row of
% zeros so that the frame is symmetrical about the centre...
Ftx = [Ftx; zeros(1, M); conj(flip(Ftx(2:end, :), 1))];

s = Ftx(:); % The transmit sequence s...

a = 2e-5; % How much the signal is attenuated by the target
L = K; % The paper uses the extended block length (K + cyclic prefix length)
fD = 400; % The doppler shift caused by the target
delay = 50; % The delay of the received signal (r) in units of samples
p = 0:L*M-1; % The p vector, as discussed in the paper


% Computing the values in the received signal r:
Z = wgn(length(s), 1, -100, 'complex'); % Complex Gaussian White noise matrix...
dopplerShift = a*exp(1i*2*pi*fD*p/L).';
r = dopplerShift.*s + Z; % The received signal

% Now to work out the correlation between s and r for different values of
% of frequency. When the correlation is high, it will indicate a likely
% target with parameters (l, fD):
nu = -800:25:800; % This is a vector of negative and positive frequency values
% Negative frequencies correspond to targets that are moving away from the
% OFDM radar. Positive frequencies are caused by targets moving towards the
% target.

% The term below is at the end of the correlation expression given in
% equation 4:
negDopplerShift = exp(-1i*2*pi.*nu.*p'/L);

% The matrix below is as follows: each column is equal to the element-wise
% multiplication between r and the corresponding column of negDopplerShift
% defined above. Each column of negDopplerShift corresponds to a complex
% exponential vector of length(p) with a different frequency, where the
% frequencies are stored inside the "nu" vector.
r_dash = r.*negDopplerShift;

% Adding in the delay by concatenating a matrix of zeros to the start of
% the r_dash matrix:
r_dash = [zeros(delay, length(nu)); r_dash];

% The vector below will store the result of the 2D correlation
% expression given in equation 4:
X = zeros(length(nu), 2*(length(r)+delay) - 1);

for j = 1:length(nu)
    X(j, :) = xcorr(r_dash(:, j), s);
end

X = (1/sqrt(L*M))*X; % Scaling, as shown in the paper...

lagVec = -(length(r)+delay)+1:length(r)+delay-1;

% Creating label variables to use with the heatmap object below:
lagLabels = string(lagVec);
lagLabels(~(mod(lagVec, 10) == 0)) = ""; % If value in vector is NOT divisble by 10, leave blank
freqLabels = string(nu);
freqLabels(~(mod(nu, 10) == 0)) = ""; % If value in vector is NOT divisble by 10, leave blank

% Plotting the heatmap:
h = heatmap(lagVec, nu, abs(X), 'Colormap', jet(300));
s = struct(h);
h.XDisplayLabels = lagLabels;
h.YDisplayLabels = freqLabels;
title('\fontsize{14}\fontname{Georgia}Correlation-based Estimation (Target with f_{\itD} = 400 Hz, \tau = 50)');
xlabel('\fontname{Georgia}\bf Delay (Samples)');
ylabel('\fontname{Georgia}\bf\itf\rm\bf (Hz)');