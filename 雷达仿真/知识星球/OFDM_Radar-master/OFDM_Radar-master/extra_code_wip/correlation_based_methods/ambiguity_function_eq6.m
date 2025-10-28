%% Computing and plotting the Ambiguity function for the OFDM waveform:
% Based on the ideas discussed in: "Comparison of Correlation-based OFDM Radar Receivers"
% Most variable names are based on those used in the above paper.
% Trying to compute equation 6 of the paper for all OFDM symbols or "blocks"
N = 32; % The number of subcarriers being used
frameSize = 128; % The frame size
M = 16; % Order of QAM scheme being used
d = (0:M-1)';

modulationType = "QPSK";

switch(modulationType)
    case "QAM"
        signalSet = qammod(d, M, 'UnitAveragePower', true); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)     
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
Vrel = 20; % The relative velocity of our target to the transmitter
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
Z = wgn(N, frameSize, -150, 'complex'); % Complex Gaussian White noise matrix...

% Next, we form the receive frame by modelling the effect of the channel
% and the target on the transmited frame:
for row = 1:N
    for column = 1:frameSize
        Frx(row, column) = Ftx(row, column)*exp(1i*2*pi*(column - 1)*To*fD)*exp(-1i*2*pi*(row - 1)*timeDelay*subcarrierSpacing) + Z(row, column);
    end
end

L = N; % Introducing L to agree with the paper...
p = 0:L-1; % The vector p, as described in the paper referenced at the top of the program
nu = -1000:50:1000; % The Greek letter "Nu", which looks like a 'v', is standing for the frequency of the E.M. wave...

% The very big matrix below will store all the
% Xm matrices:
Xm_storage = zeros(frameSize*length(nu), 2*L-1);

% This is a matrix to store the overall result of the
% ambiguity function. The values in this matrix will
% be computed using the contents of the Xm matrix.
X = zeros(length(nu), 2*L-1);

Xfinal = zeros(length(nu), 2*L-1);

% The matrix below is the result of multiplying the p vector
% by each value in the nu vector, and storing the results in
% the rows of this "p_nu_matrix" matrix. This is simply being
% used to do the computations compactly.
p_nu_matrix = p.*nu';

x = exp(-1i*2*pi*p_nu_matrix/L);

% This for loop works out Xm, as stated in the paper:
for block = 1:frameSize
    s = Ftx(:, block); % The signal s
    r = Frx(:, block); % The signal r
    
    %x2 = exp(-1i*2*pi*300*p/L);
    
    r_dash = r.*x.';
    
    % Now to calculate the correlation:
    Xm = zeros(length(nu), 2*L-1);
    
    for j = 1:length(nu)
        Xm(j, :) = xcorr(r_dash(:, j), s);
    end
    Xm = (1/sqrt(L))*Xm;
    Xm_storage(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm;
end

y = exp(-1i*2*pi*nu'.*(0:frameSize-1));

for block = 1:frameSize
    X(length(nu)*block -length(nu)+1: length(nu)*block, :) = Xm_storage(length(nu)*block -length(nu)+1: length(nu)*block, :).*y(:, block);
end

for block = 1:frameSize
    Xfinal(1:length(nu), :) = Xfinal(1:length(nu), :) + X(length(nu)*block-length(nu)+1: length(nu)*block, :);
end

%Xfinal = 1/(sqrt(frameSize))*Xfinal;
% Good work: not check 1 or 2 of the values for Xm (e.g. m = 1, m = 2 -
% frequencies of 2000 and 1000)
% x1 = exp(-1i*2*pi*-1000*p/L);
% r1 = Frx(:, 1);
% s1 = Ftx(:, 1);
% r1_dash = r1.*x1.';
% c1 = (1/sqrt(L))*xcorr(r1_dash, s1);
h = heatmap(abs(Xfinal), 'Colormap', jet(200));