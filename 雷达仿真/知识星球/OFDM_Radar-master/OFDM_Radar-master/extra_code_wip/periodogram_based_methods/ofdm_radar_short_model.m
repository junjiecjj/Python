%% Shorter model for OFDM radar:
N = 64; % The number of subcarriers being used
frameSize = 256; % The frame size
M = 16; % Order of QAM scheme being used
d = (0:M-1)';

modulationType = "QAM";

switch(modulationType)
    case "QAM"
        signalSet = qammod(d, M); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)
        %signalSet = qammod(d, M, 'UnitAveragePower', true); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)     
    case "QPSK"
        signalSet = pskmod(d, M);
    otherwise
        disp("Check the status of 'modulationType' above.")
        return % Exit the program early
end

Ftx = zeros(N/2, frameSize); % Half of the empty transmit frame

for row = 2:N/2 % We start at row 2, as row 1 is empty for the DC subcarrier...
    Ftx(row, :) = randsample(signalSet, frameSize, true);
end

% The full transmit frame. Here we are imposing Hermitian symmetry, so that
% the output from the IFFT block is real-valued...
Ftx = [Ftx; zeros(1, frameSize); conj(flip(Ftx(2:end, :), 1))];

% Index vectors: l is for the columns, k is for the rows
l = 0:frameSize-1;
k = 0:(N-1);

% OFDM system parameters:
To = 4e-6; % The OFDM symbol period
fC = 5.5e9; % The frequency of the centre subcarrier
subcarrierSpacing = 312.5e3; % The subcarrier spacing

c = 3e8; % The speed of light

% Target parameters:
Vrel = 90; % The relative velocity of our target to the transmitter
d = 50; % The distance between the target and the transmitter
targetRCS = 10; % The radar cross section of the target
targetPhi = 2*pi*rand; % The random phase offset caused by the target

% Working out the approximate Doppler shift caused when the EM wave impinges on the
% target:
fD = 2*fC*(Vrel/c);

% And the time delay:
timeDelay = 2*d/c;

% The attenuation factor, b:
b = sqrt((c*targetRCS)/((4*pi)^3*(d^4)*(fC^2)));

Frx = zeros(N, frameSize); % The received frame (currently empty)
Z = wgn(N, frameSize, -200, 'complex'); % Complex Gaussian White noise matrix...

% Next, we form the receive frame by modelling the effect of the channel
% and the target on the transmited frame:
for row = 1:N
    for column = 1:frameSize
        Frx(row, column) = b*Ftx(row, column)*exp(1i*2*pi*(column - 1)*To*fD)*exp(-1i*2*pi*(row - 1)*timeDelay*subcarrierSpacing) + Z(row, column);
    end
end

% The OFDM beacon knows what frame was sent, and hence we can divide the
% received frame by the transmitted one:
F = Frx./Ftx;

% And now we can remove the rows with "Inf" values, which were caused
% by dividing by "0"s in the transmit frame (Ftx)
F(1, :) = 0;
F(N/2 + 1, :) = 0;

% Size of FFT and IFFT calculations to be done on the rows and columns of
% the matrix F; these are also the number of rows and columns of the
% uncropped periodogram. Increasing the number of rows and columns in the
% FFT and IFFT calculations increases the resolution, but it also increases
% the number of computations needing to be done...
Nper = 4*N;
Mper = 4*frameSize;

G = 1/4; % Fraction of OFDM symbol used as a guard interval (TG/T)

% The subcarrier spacing should be at least one order of magnitude
% bigger than the largest occurring Doppler shift - hence a value of 1/10 
% for the quantity D...
D = 1/10;

Nmax = G*Nper; % The row at beyond which we should stop looking for peaks
Mmax = round(D*Mper); % The column beyond which we should stop searching for peaks

% Complex periodogram calculation:
Cper = F; % The complex periodogram is calculated using F...

% Method 1 to calculate the complex periodogram:
Cper = Cper.';
Cper = fft(Cper, Mper); % FFT of each row (length of FFT is Mper)
Cper = Cper.';
Cper = ifft(Cper, Nper); % IFFT of each column...
numCols = size(Cper, 2);
Cper = Cper(1:Nmax, :);
Cper = flip(fftshift(Cper, 2), 1);
Cper = Cper(:, (numCols/2)-Mmax:(numCols/2)+Mmax-1);

% Periodogram calculation:
Per = 1/(Nmax*(2*Mmax + 1))*(abs(Cper).^2);

% Getting the maximum value from the cropped periodogram:
maxPer = max(Per(:));

% The x and y axis are reversed in the periodogram...
[y, x] = ind2sub(size(Per), find(Per == maxPer))

% The position indexes. I'm using 'hat' in the variable name as these are
% estimates...
m_hat = x - (size(Per, 2)/2 + 1);
n_hat = size(Per, 1) - y + 1;

tau_hat = (n_hat - 1)/(Nper*subcarrierSpacing) % The time delay...

distance = (n_hat - 1)*c/(2*subcarrierSpacing*Nper)
velocity = (m_hat - 1)*c/(2*fC*To*Mper)

% The index vectors below can be used to find the distance
% and velocity values, which can be used to label the axes...
mIndexes = (1:size(Per, 2)) - (size(Per, 2)/2 + 1);
nIdexes = (1:size(Per, 1));

% Working out the corresponding distance and velocity vector
% associated with each (n, m) index...
distancesVec = (nIdexes - 1)*c/(2*subcarrierSpacing*Nper);
velocitiesVec = round((mIndexes - 1)*c/(2*fC*To*Mper));

% Creating label variables to use with the heatmap object below:
distanceLabels = string(distancesVec);
distanceLabels(~(ceil(distancesVec) == floor(distancesVec))) = "";
velocityLabels = string(velocitiesVec);
velocityLabels(~(mod(velocitiesVec, 10) == 0)) = "";

h = heatmap(velocitiesVec, flip(distancesVec), Per, 'Colormap', jet(200));
s = struct(h);
s.XAxis.TickLabelRotation = 90;
h.YDisplayLabels = flip(distanceLabels);
h.XDisplayLabels = velocityLabels;
title('\fontsize{14}\fontname{Georgia}Periodogram Target Estimation');
xlabel('\fontname{Georgia}\bf\itv\rm\bf_{rel.} (m/s)');
ylabel('\fontname{Georgia}\bfDistance (m)');