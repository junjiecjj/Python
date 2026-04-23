% Author: Oguzhan Baser
% Date: 12.01.2022
% Copyright: MIT Lisence
% https://github.com/oguzhan-baser/Channel-Estimation/tree/main
%% clear previous results
clearvars *
close all
clc
%% hyperparameters
channell = [0.74 -0.514 0.37 0.216 0.062];
snr_db=(0:2:30);
num_taps = 10;
pilot_nums = [3 5 10 20];
%% Simulations
BERs = zeros(length(pilot_nums), length(snr_db)); % initialize the BERs matrix
LSEs = zeros(length(pilot_nums), length(snr_db)); % initialize the LSEs matrix
for i = 1:length(pilot_nums) % for each pilot number
    p_nums=pilot_nums(i);
    [BERs(i,:), LSEs(i,:)]=monteCarlo(channell, snr_db, p_nums,num_taps,false);
end
[BER, ~]=monteCarlo(channell, snr_db, "",num_taps,true); % ideal channel simulation
%%
semilogy(snr_db, LSEs(2,:), '-s', 'LineWidth',2);
hold on
semilogy(snr_db, LSEs(3,:), '-s', 'LineWidth',2);
semilogy(snr_db, LSEs(4,:), '-s', 'LineWidth',2);
plotter("Least Squares Error", "Channel Error",false)
%%
figure
semilogy(snr_db, BERs(1,:), '-o', 'LineWidth',2);
hold on
semilogy(snr_db, BERs(2,:), '-o', 'LineWidth',2);
semilogy(snr_db, BERs(3,:), '-o', 'LineWidth',2);
semilogy(snr_db, BERs(4,:), '-o', 'LineWidth',2);
semilogy(snr_db, BER, '-o', 'LineWidth',2);
plotter("BER Curves", "BER", true)
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
function [originalPilots]=slidebits(bit_seq, ch_len) % a function to realize sliding bit matrix of the flipped pilots, i.o.w. X
    bit_seq = flip(bit_seq);
    row_size = length(bit_seq) + ch_len -1; % N2+M2+1
    col_size = ch_len;
    originalPilots = zeros(row_size,col_size);
    for i=1:row_size
        originalPilots(i,max(1,i-(row_size-col_size)):min(i,col_size))=bit_seq(max(end-i+1,1):end-max(i-5,0));
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
%% Equalizer
function [w]=getMMSEW(G,E,snr)
    I = eye(size(G,1));
    w = (G*G' + I./snr)\(G*E);
end
%% MONTE CARLO SIMULATION
function [BER, LSE]=monteCarlo(channel, snr_db, pilot_num,num_taps,is_channel_known) % get a boolean for MMSE to give equalizer as matrix for different SNRs
    channel_length = length(channel);
    warning off
    %%%%% SIGNAL CONSTELLATION %%%%%%%%%
    symbolBook=[1 -1];
    bitBook=[0; 1];
    nBitPerSym=size(bitBook,2);
    M=length(symbolBook);
    %%%%%%%%%%%%%% MONTE CARLO PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%
    nSymPerFrame=5000;
    max_nFrame=2000;
    fErrLim=200;
    nBitsPerFrame=nSymPerFrame*nBitPerSym;
    nBitErrors=zeros(length(snr_db), 1);
    least_squared_error_mat = zeros(length(snr_db), 1);
    nTransmittedFrames=zeros(length(snr_db), 1);
    nErroneusFrames=zeros(length(snr_db), 1);
    SYMBOLBOOK=repmat(transpose(symbolBook),1,nSymPerFrame);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for nEN = 1:length(snr_db) % SNR POINTS
        least_squared_error = 0;
        this_snr=snr_db(nEN);
        snr_pow = 10^(this_snr/10);
        sigma_noise = 1/sqrt(snr_pow);
        while (nTransmittedFrames(nEN)<max_nFrame) && (nErroneusFrames(nEN)<fErrLim)
            nTransmittedFrames(nEN) = nTransmittedFrames(nEN) + 1;
            %%%%%%%%%% INFORMATION GENERATION %%%%%%%%%%
            trSymIndices=randi(M,[1,nSymPerFrame]);
            trSymVec=symbolBook(trSymIndices);
            trBitsMat=bitBook(trSymIndices,:)';
            
            if ~ is_channel_known
            %%%%%%%%%% PILOT GENERATION %%%%%%%%%%
            trPilotIndices=randi(M,[1,pilot_num]);
            trPilotVec=symbolBook(trPilotIndices);
            original_pilots = slidebits(trPilotVec, channel_length); % generate X: sliding bits matrix with flipped pilots
            %%%%%%%%%%%%% CHANNEL %%%%%%%%%%%%%%%%%%%%
            channel_result_pilots = conv(trPilotVec, channel); % convolve the pilots with the channel
            pilot_noise = 1/sqrt(2)*[randn(1, length(channel_result_pilots))]; % generate noise for the pilot syms
            recPilotVec = channel_result_pilots+pilot_noise.*sigma_noise; %recieve the pilot symbols
            end
            
            channel_result = conv(trSymVec, channel);
            channel_result = channel_result(1:end-channel_length+1);
            trSymVec = channel_result;
            
            noise=1/sqrt(2)*[randn(1, length(trSymVec))];
            recSigVec=trSymVec+sigma_noise*noise;
            %%%%%%%%%% CHANNEL ESTIMATION %%%%%%%%%%
            if is_channel_known
                h_vec =  channel;
            else
                h_vec = (original_pilots'*original_pilots)\original_pilots'*recPilotVec'; % apply the formula derived in report
                least_squared_error=least_squared_error+sum((channel - h_vec').^2); % calculate the cumulative least squared error between estimated and original channels
            end
            %%%%%%%%%%%%% EQUALIZER %%%%%%%%%%%%%
            G = getG(h_vec, num_taps);
            E = getE(length(h_vec)+num_taps-1);
            equalizer = getMMSEW(G',E,snr_pow);
            equalizer_result = conv(recSigVec, equalizer);
            equalizer_result = equalizer_result(1:end-length(equalizer)+1);
            recSigVec = equalizer_result;
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
        if ~ is_channel_known
        least_squared_error_mat(nEN) = least_squared_error/(pilot_num*nTransmittedFrames(nEN));
        end
        sim_res=[nBitErrors nTransmittedFrames]
    end %end for (SNR points)
    disp("nBitErrors nTransmittedFrames")
    sim_res=[nBitErrors nTransmittedFrames]
    BER = nBitErrors./nTransmittedFrames/nBitsPerFrame;
    LSE = least_squared_error_mat;
end
%% figure settings
function []=plotter(titlename, yname, is_BER)
if is_BER
    legend("# Pilots= 3","# Pilots= 5", "# Pilots= 10", "# Pilots= 20", "True Channel");
else
    legend("# Pilots= 5", "# Pilots= 10", "# Pilots= 20");
end
xlabel('SNR(dB)');
ylabel(yname);
title(titlename);
grid on;
axis square;
set(gca,'FontSize',14);
ylim tight
end
%~,, |
