%% Computing and plotting the Ambiguity function for the OFDM waveform:
% Based on the ideas discussed in: "Comparison of Correlation-based OFDM Radar Receivers"
% Most variable names are based on those used in the above paper.
% Trying to compute equations 5 and 6 of the paper.
K = 32; % The number of subcarriers being used
M = 5; % The frame size
modOrder = 16; % Order of QAM scheme being used
d = (0:modOrder-1)';

modulationType = "QPSK";

switch(modulationType)
    case "QAM"
        signalSet = qammod(d, modOrder, 'UnitAveragePower', true); % Generate the signal set for the relevant QAM scheme (16, 32, 64 etc.)     
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
% the output from the IFFT block is real-valued...
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

% Now we reshape the sequence r in order to form the matrix of OFDM symbols
% or "blocks", as they are referred to in the paper:
Frx = reshape(r, K, M);

% OFDM system parameters:
To = 4e-6; % The OFDM symbol period
fC = 5.5e9; % The frequency of the centre subcarrier
subcarrierSpacing = 312.5e3; % The subcarrier spacing

c = 3e8; % The speed of light

p = 0:L-1; % The vector p, as described in the paper referenced at the top of the program
nu = -800:50:800; % The Greek letter "Nu", which looks like a 'v', is standing for the frequency of the E.M. wave...

% The very big matrix below will store all the
% Xm matrices:
Xm_storage = zeros(M*length(nu), 2*L-1);

% The matrix below is the result of multiplying the p vector
% by each value in the nu vector, and storing the results in
% the rows of this "p_nu_matrix" matrix. This is simply being
% used to do the computations compactly.
p_nu_matrix = p.*nu';

% The term below appears at the end of equation 6 in the paper:
negDopplerShift = exp(-1i*2*pi*p_nu_matrix/L);

% This for loop works out Xm, as stated in the paper:
for block = 1:M
    s = Ftx(:, block); % The signal s
    r = Frx(:, block); % The signal r
    r_dash = r.*negDopplerShift.';
    
    % Now to calculate the correlation:
    Xm = zeros(length(nu), 2*L-1);
    
    % Computing Xm by using the xcorr() function
    for j = 1:length(nu)
        Xm(j, :) = xcorr(r_dash(:, j), s);
    end
    
    Xm = (1/sqrt(K))*Xm; % Scaling, as shown in equation 6
    % Now we load this matrix for Xm into the storage matrix so it can be
    % further processed later on...
    Xm_storage(length(nu)*block-length(nu)+1: length(nu)*block, :) = Xm;
end

% By this stage we have Xm calculated for each value of m, where m is the
% index or block number - going from 0 to (M-1). Now we can work out
% equation 5:

negDopplerShift_m = exp(-1i*2*pi*nu'.*(0:M-1)); % Term used in equation 5

% This is a matrix to store the overall result of the
% ambiguity function. The values in this matrix will
% be computed using the contents of the Xm matrix.
Xm_dash = zeros(length(nu), 2*L-1);

% Check for ERRORS here:
for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    Xm_dash(startRowIndex: endRowIndex, :) = Xm_storage(startRowIndex: endRowIndex, :).*negDopplerShift_m(:, block);
end

% This matrix below will store the final result of the ambiguity function
% computation:
X = zeros(length(nu), 2*L-1);

% Check for ERRORS here:
for block = 1:M
    startRowIndex = length(nu)*block - length(nu) + 1;
    endRowIndex = length(nu)*block;
    X(1:length(nu), :) = X(1:length(nu), :) + Xm_dash(startRowIndex: endRowIndex, :);
end

X = (1/(sqrt(M)))*X;
delayVec = -K+1:K-1;
h = heatmap(delayVec, nu, abs(X), 'Colormap', jet(300));
title('\fontsize{14}\fontname{Georgia}Correlation-based Estimation (Target f_{\itD} = 400 Hz');
xlabel('\fontname{Georgia}\bf Delay (Samples)');
ylabel('\fontname{Georgia}\bf\itf\rm\bf (Hz)');