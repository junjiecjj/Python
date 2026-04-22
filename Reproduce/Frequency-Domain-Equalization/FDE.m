% Author: Oguzhan Baser
% Date: 12.01.2022
% Copyright: MIT Lisence

%% clear previous results
clearvars *
close all
clc
%% hyperparameters
channell = [0.74 -0.514 0.37 0.216 0.062];
snr_db=(0:2:20);
cyclicPrefixes = [1 3 5 10];
NFFT = 1000; % num of FFT or num of symbols in each frame
num_taps = 10;
%% Simulations
% FDEs
for cprefix = cyclicPrefixes
    FDE_simulate(channell, snr_db, "MMSE FDE, CP= "+num2str(cprefix), cprefix, NFFT)
end
% TDE
MMSE_simulate(num_taps, channell, snr_db, "MMSE TDE")
plotter() % figure settings

%% Simulation Functions
function []=FDE_simulate(channell, snr_db, name, cyclicPrefixes, NFFTs)
    [BER]=monteCarlo(channell,"", snr_db, true, "", "", name, cyclicPrefixes, NFFTs, "FDE "+num2str(cyclicPrefixes));
end
function []=MMSE_simulate(num_taps, channell, snr_db, name)
    G = getG(channell, num_taps);
    E = getE(length(channell)+num_taps-1);
    disp("G mat is: ")
    disp(G)
    disp("E mat is: ")
    disp(E)
    w = getMMSEW(G',E,1);
    disp("w for 0 SNR is: ")
    disp(w)
    [BER]=monteCarlo(channell, w, snr_db, false, G, E, name, "", "", "TDE");
end
%% get G matrix
function [G] = getG(channel, num_tap)
    row_size = length(channel) + num_tap -1 ; % N2+M2+1
    col_size = num_tap;
    G = zeros(row_size, col_size);
    channel_flipped=flip(channel);
    for row=1:row_size
        [start_pt, end_pt] = get_idx(row, length(channel)-1, col_size-1);
        channel_to_be_inserted = channel_flipped(end-(end_pt-start_pt):end);
        if row>col_size
            channel_to_be_inserted = channel_flipped(1:end_pt-start_pt+1);
        end
        G(row,start_pt:end_pt)=channel_to_be_inserted;
    end
end
%% get starting and ending point of the flowing channel in G
function [start_pt, end_pt] = get_idx(row, N1plusN2, M1plusM2)
    end_pt = row;
    start_pt = 1;
    if row>M1plusM2+1
        end_pt = M1plusM2+1;
    end
    if row>N1plusN2+1
        start_pt = row-N1plusN2;
    end
end
%% get e vector
function [E] = getE(Esize)
    E = zeros(Esize,1);
    E(1,1)=1;
end
%% Equalizers
function [w]=getMMSEW(G,E,snr)
    I = eye(size(G,1));
    w = (G*G' + I./snr)\(G*E);
end
function [w]=getMMSEFDEW(channel,NFFT,sigma_noise)
    w = 1./(fft(channel, NFFT) + (sigma_noise)^2); % implements the formula dicussed in the report
end
%% MONTE CARLO SIMULATION
function [BER]=monteCarlo(channel, equalizer, snr_db, is_FDE, G, E, name, cyclicPrefix, NFFT, titlee) % get a boolean for MMSE to give equalizer as matrix for different SNRs
    channel_length = length(channel);
    equalizer_length = length(equalizer);
    warning off
    %%%%% SIGNAL CONSTELLATION %%%%%%%%%
    symbolBook=[1 -1];
    bitBook=[0; 1];
    nBitPerSym=size(bitBook,2);
    M=length(symbolBook);
    %%%%%%%%%%%%%% MONTE CARLO PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%
    if is_FDE
        nSymPerFrame=NFFT;
    else
        nSymPerFrame=1000;
    end
    nBitsPerFrame=nSymPerFrame*nBitPerSym;
    max_nFrame=2000;
    fErrLim=200;
    nBitErrors=zeros(length(snr_db), 1);
    nTransmittedFrames=zeros(length(snr_db), 1);
    nErroneusFrames=zeros(length(snr_db), 1);
    SYMBOLBOOK=repmat(transpose(symbolBook),1,nSymPerFrame);
    switch_tictoc = true;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for nEN = 1:length(snr_db) % SNR POINTS
        this_snr=snr_db(nEN);
        snr_pow = 10^(this_snr/10);
        sigma_noise = 1/sqrt(snr_pow);
        if is_FDE
            equalizer=getMMSEFDEW(channel,NFFT,sigma_noise);
        end
        while (nTransmittedFrames(nEN)<max_nFrame) && (nErroneusFrames(nEN)<fErrLim)
            nTransmittedFrames(nEN) = nTransmittedFrames(nEN) + 1;
            %%%%%%%%%% INFORMATION GENERATION %%%%%%%%%%
            trSymIndices=randi(M,[1,nSymPerFrame]);
            trSymVec=symbolBook(trSymIndices);
            trBitsMat=bitBook(trSymIndices,:)';
            if is_FDE % addition of the cyclic prefix to enable circular convolution operation
                trSymVec = [trSymVec(end-cyclicPrefix+1:end) trSymVec];
            end
            %%%%%%%%%%%%% CHANNEL %%%%%%%%%%%%%%%%%%%%
            channel_result = conv(trSymVec, channel);
            channel_result = channel_result(1:end-channel_length+1);
            trSymVec = channel_result;
            noise=1/sqrt(2)*[randn(1, length(trSymVec)) + 1j*randn(1,length(trSymVec))];
            recSigVec=trSymVec+sigma_noise*noise;
            %%%%%%%%%%%%% EQUALIZER %%%%%%%%%%%%%
            if switch_tictoc % only once
                disp("*************New Simulation : "+ titlee + " ********")
                tic % start to measure equalization time
            end
            if is_FDE 
                recSigVec = recSigVec(1+cyclicPrefix: end); % deletion of cyclic prefix
                recSigVec = fft(recSigVec, NFFT); % taking FFT of incoming signal
                recSigVec = recSigVec.*equalizer; % equalizing with multiplication
                recSigVec = ifft(recSigVec, NFFT); % IFFT
            else
                equalizer = getMMSEW(G',E,snr_pow); % get time domain MMSE 10-tap equalizer
                equalizer_result = conv(recSigVec, equalizer);
                equalizer_result = equalizer_result(1:end-equalizer_length+1); % get causal conv output
                recSigVec = equalizer_result;
            end
            if switch_tictoc % only once
                toc % stop to measure equalization time
                switch_tictoc = false;
            end
            %%%%%%%%%%%%% DETECTOR %%%%%%%%%%%%%
            RECSIGVEC=repmat(recSigVec,length(symbolBook),1);
            distance_mat=abs(SYMBOLBOOK-RECSIGVEC);
            [~, det_sym_ind]=min(distance_mat,[],1);
            detected_bits=[bitBook(det_sym_ind, :)]';
            err = sum(sum(abs(trBitsMat-detected_bits)));
            nBitErrors(nEN)=nBitErrors(nEN)+err;
            if err~=0
                nErroneusFrames(nEN)=nErroneusFrames(nEN)+1;
            end
        end % End of while loop
        sim_res=[nBitErrors nTransmittedFrames];
    end %end for (SNR points)
    disp("nBitErrors nTransmittedFrames")
    sim_res=[nBitErrors nTransmittedFrames];
    disp(sim_res)
    BER = nBitErrors./nTransmittedFrames/nBitsPerFrame;
    semilogy(snr_db, BER, '-o', 'LineWidth',2);
    hold on
    grid on
    title(name)
    xlabel("SNR")
    ylabel("BER")
end
%% figure settings
function []=plotter()
legend('FDE CP=1','FDE CP=3','FDE CP=5','FDE CP=10', 'TDE Taps=10');
xlabel('SNR(dB)');
ylabel('BER');
title('MMSE');
grid on;
axis square;
set(gca,'FontSize',14);
ylim tight
end
%~,, |