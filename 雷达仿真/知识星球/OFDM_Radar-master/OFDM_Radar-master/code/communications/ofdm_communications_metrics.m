%% Investigating the relationship between capacity, bandwidth, and SNR:
%
%
% Author: Dylan Boland (Student)

IEEEstandard = "802.11a"; % the IEEE standard that we want to simulate...

% Let us define the OFDM system parameters based on the IEEE 802 standard
% in use:
switch (IEEEstandard)
    case "802.11a"
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11a
        Tu = 1/subcarrierSpacing; % the "useful" symbol period
        G = 1/4; % the guard interval fraction
        Tg = G*Tu; % the guard interval time
        To = Tu + Tg; % the OFDM symbol duration (period)
        fC = 5.5e9; % 802.11a uses the 5 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used
        M = 7; % no. of OFDM symbols in a frame or burst 
    case "802.11g"
        subcarrierSpacing = 312.5e3; % subcarrier spacing for 802.11g
        Tu = 1/subcarrierSpacing; % the "useful" symbol period
        G = 1/4; % the guard interval fraction
        Tg = G*Tu; % the guard interval time
        To = Tu + Tg; % the OFDM symbol duration (period)
        fC = 2.4e9; % 802.11a uses the 2.4 GHz band
        N = 52; % 48 for data, and 4 pilot tones
        numPilotTones = 4; % no. pilot tones used
        M = 7; % no. of OFDM symbols in a frame or burst        
end

modType = "QAM";
modOrder = 64;
r = 3/4; % the code rate
bitsPerSymbol = log2(modOrder);
bitsPerOFDMsymbol = (N-numPilotTones)*bitsPerSymbol*r;
dataRate = (bitsPerOFDMsymbol/To)/10^6; % the data rate in Mbits/s
bandwidth = N*subcarrierSpacing; % the bandwidth of the system...