clc
clear
dbstop if error

% --- Parallel Pool Setup ---
% poolobj = gcp('nocreate');
% if ~isempty(poolobj)
%     delete(poolobj);
% end
% Consider starting the pool later if needed, e.g., before parfor
% gcp;

% --- Simulation Flags ---
MMSE_flag = 1;      % 1 to enable MMSE equalization
LDPC_flag = 1;      % 1 to ENABLE LDPC, 0 to DISABLE LDPC (uncoded)

% --- Simulation Parameters ---
max_itr = 1000;     % Number of iterations (Adjust as needed)

% --- Basic Modulation Parameters ---
mod_order = 4; % QPSK
bitsPerSymbol = log2(mod_order);

% --- Determine Nominal Frame Size (for consistent block structure) ---
% Use a typical/default LDPC setting to establish frame symbol length
% This helps keep sub-block processing consistent regardless of LDPC flag
nominalSymbolLength = [];
try
    fprintf('Determining nominal frame size using default LDPC Rate 1/2...\n');
    % Use a common DVB-S.2 rate like 1/2 to get N=64800
    H_setup = dvbs2ldpc(1/2, 'sparse');
    [~, N_setup] = size(H_setup);
    if mod(N_setup, bitsPerSymbol) == 0
        nominalSymbolLength = N_setup / bitsPerSymbol;
        fprintf('Nominal frame size based on Rate 1/2 LDPC: N=%d -> %d symbols\n', ...
                N_setup, nominalSymbolLength);
    else
         warning('Default LDPC N=%d not multiple of bitsPerSymbol=%d. Using fallback size.', N_setup, bitsPerSymbol);
    end
catch ME
    fprintf('Warning: Could not get default LDPC params for sizing. Error: %s.\n', ME.message);
    if contains(ME.identifier, 'comm:dvbs2ldpc')
        fprintf('Ensure Communications Toolbox with DVB-S.2 standard support is installed.\n');
    end
end

if isempty(nominalSymbolLength)
    nominalSymbolLength = 32400; % Fallback nominal symbol length (e.g., 64800 bits / 2 bits/sym)
    fprintf('Using fallback nominal frame size: %d symbols\n', nominalSymbolLength);
end
clear H_setup N_setup; % Clean up temporary setup variables

% --- Waveform & Block Processing Parameters (Based on Nominal Size) ---
fullSymbolLength = nominalSymbolLength; % Base frame size on nominal length
subBlockLength = 1024; % Processing block size
if mod(fullSymbolLength, subBlockLength) ~= 0
    fprintf('Warning: Nominal symbol length %d is not an integer multiple of sub-block length %d.\n', ...
            fullSymbolLength, subBlockLength);
    % Option: Adjust subBlockLength or nominalSymbolLength if needed, or accept padding.
end
numSubBlocks = ceil(fullSymbolLength / subBlockLength);
paddedFullLength = numSubBlocks * subBlockLength; % Length after padding
numPadSymbols = paddedFullLength - fullSymbolLength; % Symbols to pad

% --- LDPC Configuration (Conditional) ---
if LDPC_flag == 1
    fprintf('LDPC Coding ENABLED.\n');
    ldpcCodeRate = 1/2; % Target code rate (must be supported by dvbs2ldpc)
    fprintf('Target DVB-S.2 Code Rate: %.2f\n', ldpcCodeRate);

    try
        fprintf('Attempting to configure LDPC using dvbs2ldpc(Rate, ''sparse'')...\n');
        H = dvbs2ldpc(ldpcCodeRate, 'sparse');
        if isempty(H)
             error('comm:dvbs2ldpc:GeneratedEmptyMatrix', 'dvbs2ldpc returned an empty matrix for rate %.2f.', ldpcCodeRate);
        end

        % Derive K and N from the H matrix dimensions
        [numParityBits, codeLength] = size(H); % N = number of columns
        infoLength = codeLength - numParityBits; % K = N - M

        fprintf('LDPC Setup Success (DVB-S.2): Rate=%.2f -> K=%d, N=%d (Fixed by standard)\n', ldpcCodeRate, infoLength, codeLength);

        % Configure encoder and decoder using the obtained Parity Check Matrix H
        cfgLDPCEnc = ldpcEncoderConfig(H);
        cfgLDPCDec = ldpcDecoderConfig(H);
        ldpcMaxIter = 50; % Max iterations for LDPC decoder
        actualRate = infoLength / codeLength; % Actual code rate
        bitsPerFrame = codeLength; % Total bits in the coded frame

        % --- Consistency Check: Adjust frame size if actual LDPC N differs ---
        actualFrameSymbols = codeLength / bitsPerSymbol;
        if mod(codeLength, bitsPerSymbol) ~= 0
            error('Actual LDPC codeLength (%d) must be a multiple of bitsPerSymbol (%d)', codeLength, bitsPerSymbol);
        end
        if actualFrameSymbols ~= fullSymbolLength
             fprintf('Info: Actual LDPC requires %d symbols/frame. Overriding nominal size %d.\n', ...
                     actualFrameSymbols, fullSymbolLength);
             fullSymbolLength = actualFrameSymbols;
             numSubBlocks = ceil(fullSymbolLength / subBlockLength);
             paddedFullLength = numSubBlocks * subBlockLength;
             numPadSymbols = paddedFullLength - fullSymbolLength;
        end

    catch ME
        fprintf('Error configuring LDPC using dvbs2ldpc: %s\n', ME.message);
        fprintf('Identifier: %s\n', ME.identifier);
        % ... (rest of specific error messages from original code) ...
        error('Failed to configure LDPC parameters. Halting simulation.');
    end

else % LDPC_flag == 0
    fprintf('LDPC Coding DISABLED (Uncoded simulation).\n');
    % For uncoded simulation, infoLength = codeLength = bitsPerFrame
    bitsPerFrame = fullSymbolLength * bitsPerSymbol; % Total bits based on nominal frame size
    infoLength = bitsPerFrame; % Number of info bits = total bits in frame
    codeLength = bitsPerFrame; % Effective code length is the same
    actualRate = 1.0; % Uncoded rate is 1
    % Set placeholders for LDPC config vars
    H = []; cfgLDPCEnc = []; cfgLDPCDec = []; ldpcMaxIter = 0;
    fprintf('Uncoded configuration: K = N = %d bits per frame.\n', bitsPerFrame);
end

fprintf('Final Frame Configuration: %d Symbols/Frame (%d padded), %d Bits/Frame, Rate=%.4f\n', ...
        fullSymbolLength, paddedFullLength, bitsPerFrame, actualRate);
fprintf('Block Processing: %d Sub-blocks of size %d symbols.\n', numSubBlocks, subBlockLength);


% --- Channel Parameters ---
multiPathNum = 2;    % Number of multipaths per satellite
satNum = 3;          % Number of satellites
delayMax = 20;       % Max delay in samples (l_max) relative to subBlockLength Ts
cpLength = delayMax; % Conceptual CP length

% --- Timing and Doppler Parameters ---
sampleRate = 133.3e6; % Sample rate in Hz
Ts = 1/sampleRate;      % Sample time
doppler_resolution = 1/(subBlockLength*Ts);
fd_max = 0.398e6; % Max Doppler shift in Hz
dopplerMax = fd_max/doppler_resolution; % Max Doppler index k_max, relative to subBlockLength duration

% --- SNR Setup ---
EBN0 = [0:2:10]; % Eb/N0 range in dB (Adjust as needed, lower values useful for uncoded)
% Eb/N0 is Energy per INFORMATION bit / Noise PSD
SNR_dB = EBN0 + 10*log10(actualRate * bitsPerSymbol); % SNR per symbol (Es/N0)
SNR = 10.^(SNR_dB/10); % SNR in linear scale
SymbolEnergy = 1; % Assuming normalized symbol energy after modulation
NoiseVariance = SymbolEnergy ./ SNR; % Noise variance sigma^2
NoiseCoe = sqrt(NoiseVariance); % Noise standard deviation sigma

fprintf('Simulation Parameters:\n');
fprintf('  LDPC Enabled: %d\n', LDPC_flag);
fprintf('  MMSE Enabled: %d\n', MMSE_flag);
fprintf('  Modulation: %d-PSK\n', mod_order);
fprintf('  Sub-Block Length (N_sub): %d\n', subBlockLength);
fprintf('  Max Doppler Index (k_max): %d\n', dopplerMax);
fprintf('  Max Delay Index (l_max): %d\n', delayMax);
fprintf('  Eb/N0 Range (dB): %s\n', mat2str(EBN0));
fprintf('  SNR Range (dB): %s\n', mat2str(SNR_dB, 3));


% --- Create Transformation Matrices (Dimensions based on subBlockLength) ---
% Check if createMatrix needs adjustment (e.g., OFDM output swap)
[DFT_sub, IDFT_sub] = createMatrix(subBlockLength, delayMax, dopplerMax, 'OFDM');
[DFnT_sub, IDFnT_sub] = createMatrix(subBlockLength, delayMax, dopplerMax, 'OCDM');
[DAFT_sub, IDAFT_sub] = createMatrix(subBlockLength, delayMax, dopplerMax, 'AFDM');
[SFFT_sub, ISFFT_sub] = createMatrix(subBlockLength, delayMax, dopplerMax, 'OTFS'); % Verify OTFS requires N=M^2 if used

% --- Initialize Result Storage ---
numErrorsSCinTotal_MMSE   = zeros(length(SNR_dB),1);
numErrorsOFDMinTotal_MMSE = zeros(length(SNR_dB),1);
numErrorsOCDMinTotal_MMSE = zeros(length(SNR_dB),1);
numErrorsAFDMinTotal_MMSE = zeros(length(SNR_dB),1);
numErrorsOTFSinTotal_MMSE = zeros(length(SNR_dB),1);
berSC_MMSE   = zeros(1, length(SNR_dB));
berOFDM_MMSE = zeros(1, length(SNR_dB));
berOCDM_MMSE = zeros(1, length(SNR_dB));
berAFDM_MMSE = zeros(1, length(SNR_dB));
berOTFS_MMSE = zeros(1, length(SNR_dB));

% --- Main Simulation Loop ---
startTime = tic;
for iteration_time = 1: max_itr

    % --- Generate Source Bits ---
    % Size depends on whether LDPC is used (infoLength) or not (bitsPerFrame)
    data_bits = randi([0 1], infoLength, 1); % Correct size based on LDPC_flag check earlier

    % --- Encoding (Conditional) ---
    if LDPC_flag == 1
        encoded_bits = ldpcEncode(data_bits, cfgLDPCEnc);
    else
        encoded_bits = data_bits; % No encoding, bits to modulate = info bits
    end
    % encoded_bits should now always have size 'codeLength' (where codeLength=infoLength if LDPC_flag=0)
    % Ensure codeLength correctly matches fullSymbolLength * bitsPerSymbol

    % --- Modulation ---
    modulatedDataFull_unnorm = pskmod(encoded_bits, mod_order, pi/mod_order, 'InputType', 'bit');
    actualPower = mean(abs(modulatedDataFull_unnorm).^2);
    modulatedDataFull = modulatedDataFull_unnorm / sqrt(actualPower); % Normalize power
    modulatedDataFull = modulatedDataFull(:); % Ensure column vector

    % --- Padding ---
    if numPadSymbols > 0
        % Pad with zeros (or other known symbols if needed for channel est/sync)
        modulatedDataPadded = [modulatedDataFull; zeros(numPadSymbols, 1)];
    else
        modulatedDataPadded = modulatedDataFull;
    end
    if length(modulatedDataPadded) ~= paddedFullLength
       error('Padding error: Padded data length %d != expected %d', length(modulatedDataPadded), paddedFullLength);
    end


    % --- Initialize temporary error storage for parallel SNR loop ---
    err_SC_tmp   = zeros(length(SNR_dB),1);
    err_OFDM_tmp = zeros(length(SNR_dB),1);
    err_OCDM_tmp = zeros(length(SNR_dB),1);
    err_AFDM_tmp = zeros(length(SNR_dB),1);
    err_OTFS_tmp = zeros(length(SNR_dB),1);

    % --- Process per SNR point in parallel ---
    for snr_pin = 1:length(SNR_dB) % Use 'parfor' if functions are compatible

        current_snr_lin = SNR(snr_pin);
        current_noise_var = NoiseVariance(snr_pin); % Noise variance sigma^2

        % --- Generate INITIAL physical channel parameters for this SNR worker thread ---
        % These parameters evolve across sub-blocks within this worker
        numTotalPaths = multiPathNum * satNum;
        workerGains_init = (randn(numTotalPaths, 1) + 1i * randn(numTotalPaths, 1)) / sqrt(2); % Normalize later
        workerDelays_init = randi([0 delayMax], numTotalPaths, 1);
        workerDopplersHz_init = (rand(numTotalPaths, 1) - 0.5) * 2 * fd_max; % Doppler in Hz
        % Apply normalization consistent with createMultiSatChannel... (if function exists)
        % Example: Normalize total power per satellite path set or overall
        chanNormFactor = sqrt(sum(abs(workerGains_init).^2)); % Example: Normalize total power to 1
         if chanNormFactor > 1e-9
             workerGains_init = workerGains_init / chanNormFactor;
         end
        % Or use per-satellite normalization if appropriate model used in constructHeffFromParams

        % --- Initialize channel state FOR THIS SNR WORKER THREAD ---
        workerGains = workerGains_init;
        workerDelays = workerDelays_init;
        workerDopplersHz = workerDopplersHz_init;

        % --- Preallocate storage for equalized signals from all sub-blocks ---
        eqSignalSubBlocks_SC   = zeros(paddedFullLength, 1);
        eqSignalSubBlocks_OFDM = zeros(paddedFullLength, 1);
        eqSignalSubBlocks_OCDM = zeros(paddedFullLength, 1);
        eqSignalSubBlocks_AFDM = zeros(paddedFullLength, 1);
        eqSignalSubBlocks_OTFS = zeros(paddedFullLength, 1);

        % --- Loop through sub-blocks ---
        for iSubBlock = 1:numSubBlocks
            % Get sub-block data (modulated symbols)
            startIndex = (iSubBlock-1)*subBlockLength + 1;
            endIndex = iSubBlock*subBlockLength;
            subBlockModData = modulatedDataPadded(startIndex:endIndex);

            % --- Apply Waveforms (Inverse Transforms) ---
            scDataSub   = subBlockModData;
            ofdmDataSub = IDFT_sub * subBlockModData;
            ocdmDataSub = IDFnT_sub * subBlockModData;
            afdmDataSub = IDAFT_sub * subBlockModData;
            otfsDataSub = ISFFT_sub * subBlockModData;

            % --- Generate Heff_sub for THIS sub-block ---
            Heff_sub = constructHeffFromParams(subBlockLength, workerGains, workerDelays, workerDopplersHz, Ts);

            % --- Apply Channel ---
            channeledSubBlock_SC   = Heff_sub * scDataSub;
            channeledSubBlock_OFDM = Heff_sub * ofdmDataSub;
            channeledSubBlock_OCDM = Heff_sub * ocdmDataSub;
            channeledSubBlock_AFDM = Heff_sub * afdmDataSub;
            channeledSubBlock_OTFS = Heff_sub * otfsDataSub;

            % --- Add Noise ---
            noiseSubBlock = sqrt(current_noise_var/2) * (randn(subBlockLength, 1) + 1i * randn(subBlockLength, 1));
            rxSubBlock_SC   = channeledSubBlock_SC   + noiseSubBlock;
            rxSubBlock_OFDM = channeledSubBlock_OFDM + noiseSubBlock;
            rxSubBlock_OCDM = channeledSubBlock_OCDM + noiseSubBlock;
            rxSubBlock_AFDM = channeledSubBlock_AFDM + noiseSubBlock;
            rxSubBlock_OTFS = channeledSubBlock_OTFS + noiseSubBlock;

            % --- MMSE Equalizer Calculation & Application ---
            if MMSE_flag == 1
                % W = (H'*H + sigma^2*I)^(-1) * H'
                W_sub = (Heff_sub'*Heff_sub + eye(subBlockLength) * current_noise_var) \ Heff_sub';
                eqSignalSub_SC   = W_sub * rxSubBlock_SC;
                eqSignalSub_OFDM = W_sub * rxSubBlock_OFDM;
                eqSignalSub_OCDM = W_sub * rxSubBlock_OCDM;
                eqSignalSub_AFDM = W_sub * rxSubBlock_AFDM;
                eqSignalSub_OTFS = W_sub * rxSubBlock_OTFS;
            else % No equalization (or just identity matrix)
                eqSignalSub_SC   = rxSubBlock_SC;
                eqSignalSub_OFDM = rxSubBlock_OFDM;
                eqSignalSub_OCDM = rxSubBlock_OCDM;
                eqSignalSub_AFDM = rxSubBlock_AFDM;
                eqSignalSub_OTFS = rxSubBlock_OTFS;
            end

            % --- Store equalized signal BEFORE forward transform ---
            eqSignalSubBlocks_SC(startIndex:endIndex)   = eqSignalSub_SC;
            eqSignalSubBlocks_OFDM(startIndex:endIndex) = eqSignalSub_OFDM;
            eqSignalSubBlocks_OCDM(startIndex:endIndex) = eqSignalSub_OCDM;
            eqSignalSubBlocks_AFDM(startIndex:endIndex) = eqSignalSub_AFDM;
            eqSignalSubBlocks_OTFS(startIndex:endIndex) = eqSignalSub_OTFS;

            % --- Evolve physical channel parameters for the NEXT sub-block ---
            subBlockDuration = subBlockLength * Ts;
            [workerGains, workerDelays, workerDopplersHz] = evolveChannelParams(workerGains, workerDelays, workerDopplersHz, subBlockDuration);

        end % End sub-block loop (iSubBlock)

        % --- Post Sub-block Processing (Apply Forward Transforms) ---
        % Transform the assembled, equalized *waveform domain* signals back to *modulation domain* symbols
        eqSymbolsFull_SC   = zeros(fullSymbolLength, 1); % Stores symbols after forward transform
        eqSymbolsFull_OFDM = zeros(fullSymbolLength, 1);
        eqSymbolsFull_OCDM = zeros(fullSymbolLength, 1);
        eqSymbolsFull_AFDM = zeros(fullSymbolLength, 1);
        eqSymbolsFull_OTFS = zeros(fullSymbolLength, 1);

        for iSubBlock = 1:numSubBlocks
             startIndexSub = (iSubBlock-1)*subBlockLength + 1;
             endIndexSub = iSubBlock*subBlockLength; % Index into padded vectors

             % Index for storing into the final non-padded symbol vector
             startIndexSym = (iSubBlock-1)*subBlockLength + 1;
             endIndexSym = min(iSubBlock*subBlockLength, fullSymbolLength);
             numSymsInThisBlock = endIndexSym - startIndexSym + 1;

             % Apply forward transform to the corresponding segment of equalized waveform signal
             syms_OFDM = DFT_sub  * eqSignalSubBlocks_OFDM(startIndexSub:endIndexSub);
             syms_OCDM = DFnT_sub * eqSignalSubBlocks_OCDM(startIndexSub:endIndexSub);
             syms_AFDM = DAFT_sub * eqSignalSubBlocks_AFDM(startIndexSub:endIndexSub);
             syms_OTFS = SFFT_sub * eqSignalSubBlocks_OTFS(startIndexSub:endIndexSub);

             % Store the valid symbols (handling potential last partial block)
             eqSymbolsFull_SC(startIndexSym:endIndexSym)   = eqSignalSubBlocks_SC(startIndexSub : startIndexSub+numSymsInThisBlock-1); % SC needs no transform
             eqSymbolsFull_OFDM(startIndexSym:endIndexSym) = syms_OFDM(1:numSymsInThisBlock);
             eqSymbolsFull_OCDM(startIndexSym:endIndexSym) = syms_OCDM(1:numSymsInThisBlock);
             eqSymbolsFull_AFDM(startIndexSym:endIndexSym) = syms_AFDM(1:numSymsInThisBlock);
             eqSymbolsFull_OTFS(startIndexSym:endIndexSym) = syms_OTFS(1:numSymsInThisBlock);
        end

        % --- Soft Demodulation (LLR Calculation) ---
        % Use noise variance appropriate for LLR calculation (approximation after MMSE)
        llrNoiseVar = current_noise_var; % Common approximation
        % Alternative: Try to estimate post-MMSE noise variance if needed (more complex)

        llrFull_SC   = pskdemod(eqSymbolsFull_SC,   mod_order, pi/mod_order, 'OutputType','approxllr', 'NoiseVariance', llrNoiseVar);
        llrFull_OFDM = pskdemod(eqSymbolsFull_OFDM, mod_order, pi/mod_order, 'OutputType','approxllr', 'NoiseVariance', llrNoiseVar);
        llrFull_OCDM = pskdemod(eqSymbolsFull_OCDM, mod_order, pi/mod_order, 'OutputType','approxllr', 'NoiseVariance', llrNoiseVar);
        llrFull_AFDM = pskdemod(eqSymbolsFull_AFDM, mod_order, pi/mod_order, 'OutputType','approxllr', 'NoiseVariance', llrNoiseVar);
        llrFull_OTFS = pskdemod(eqSymbolsFull_OTFS, mod_order, pi/mod_order, 'OutputType','approxllr', 'NoiseVariance', llrNoiseVar);

        % --- Decoding or Hard Decision (Conditional) ---
        if LDPC_flag == 1
            % LDPC Decoding
            decBits_SC   = ldpcDecode(llrFull_SC(:),   cfgLDPCDec, ldpcMaxIter);
            decBits_OFDM = ldpcDecode(llrFull_OFDM(:), cfgLDPCDec, ldpcMaxIter);
            decBits_OCDM = ldpcDecode(llrFull_OCDM(:), cfgLDPCDec, ldpcMaxIter);
            decBits_AFDM = ldpcDecode(llrFull_AFDM(:), cfgLDPCDec, ldpcMaxIter);
            decBits_OTFS = ldpcDecode(llrFull_OTFS(:), cfgLDPCDec, ldpcMaxIter);
            numComparedBits = infoLength; % Compare only the original info bits
            originalBits = data_bits;     % Original info bits for comparison
        else
            % Hard Decision for Uncoded BER
            decBits_SC   = double(llrFull_SC(:) < 0);
            decBits_OFDM = double(llrFull_OFDM(:) < 0);
            decBits_OCDM = double(llrFull_OCDM(:) < 0);
            decBits_AFDM = double(llrFull_AFDM(:) < 0);
            decBits_OTFS = double(llrFull_OTFS(:) < 0);
            % Ensure decBits vectors have length 'codeLength' (== bitsPerFrame)
            numComparedBits = codeLength; % Compare all transmitted bits
            originalBits = data_bits;     % Original transmitted bits (same as info bits here)
        end

        % --- Error Calculation ---
        % Ensure lengths match before comparison
         if length(decBits_SC) < numComparedBits || length(originalBits) < numComparedBits
              fprintf('Error in parfor worker: Decoded bits length (%d) or original bits length (%d) mismatch comparison length (%d)\n', ...
                      length(decBits_SC), length(originalBits), numComparedBits);
              % Handle error appropriately, e.g., assign NaN or skip accumulation
              err_SC_tmp(snr_pin) = NaN; err_OFDM_tmp(snr_pin) = NaN; % etc.
         else
             err_SC_tmp(snr_pin)   = sum(decBits_SC(1:numComparedBits)   ~= originalBits(1:numComparedBits));
             err_OFDM_tmp(snr_pin) = sum(decBits_OFDM(1:numComparedBits) ~= originalBits(1:numComparedBits));
             err_OCDM_tmp(snr_pin) = sum(decBits_OCDM(1:numComparedBits) ~= originalBits(1:numComparedBits));
             err_AFDM_tmp(snr_pin) = sum(decBits_AFDM(1:numComparedBits) ~= originalBits(1:numComparedBits));
             err_OTFS_tmp(snr_pin) = sum(decBits_OTFS(1:numComparedBits) ~= originalBits(1:numComparedBits));
         end

    end % End parfor snr_pin

    % --- Accumulate errors from parallel loop (handle potential NaNs) ---
    numErrorsSCinTotal_MMSE   = numErrorsSCinTotal_MMSE   + err_SC_tmp;
    numErrorsOFDMinTotal_MMSE = numErrorsOFDMinTotal_MMSE + err_OFDM_tmp;
    numErrorsOCDMinTotal_MMSE = numErrorsOCDMinTotal_MMSE + err_OCDM_tmp;
    numErrorsAFDMinTotal_MMSE = numErrorsAFDMinTotal_MMSE + err_AFDM_tmp;
    numErrorsOTFSinTotal_MMSE = numErrorsOTFSinTotal_MMSE + err_OTFS_tmp;

    % --- Display Progress ---
    if mod(iteration_time, 10) == 0 || iteration_time == max_itr % Display less frequently
        elapsedTime = toc(startTime);
        estRemaining = (elapsedTime / iteration_time) * (max_itr - iteration_time);
        fprintf('Iteration %d/%d (Elapsed: %.1fs, Est. Remaining: %.1fs)\n', ...
                 iteration_time, max_itr, elapsedTime, estRemaining);

        % Calculate current BER based on the number of bits compared in this configuration
        if LDPC_flag == 1
             numComparedBitsPerIter = infoLength;
        else
             numComparedBitsPerIter = codeLength;
        end
        totalComparedBitsSoFar = numComparedBitsPerIter * iteration_time;

        % Avoid division by zero if no bits compared yet
        if totalComparedBitsSoFar > 0
             current_berSC   = numErrorsSCinTotal_MMSE(end) / totalComparedBitsSoFar;
             current_berOFDM = numErrorsOFDMinTotal_MMSE(end) / totalComparedBitsSoFar;
             current_berOCDM = numErrorsOCDMinTotal_MMSE(end) / totalComparedBitsSoFar;
             current_berAFDM = numErrorsAFDMinTotal_MMSE(end) / totalComparedBitsSoFar;
             current_berOTFS = numErrorsOTFSinTotal_MMSE(end) / totalComparedBitsSoFar;

             fprintf('  Current BER (MMSE) @ max EbN0 (%.1f dB):\n', EBN0(end));
             fprintf('    SC: %.2e | OFDM: %.2e | OCDM: %.2e | AFDM: %.2e | OTFS: %.2e\n', ...
                  current_berSC, current_berOFDM, current_berOCDM, current_berAFDM, current_berOTFS);
        end
    end
end % End for iteration_time loop

% --- Final BER Calculation ---
actual_iterations = iteration_time;
if LDPC_flag == 1
    numComparedBitsPerIter = infoLength;
else
    numComparedBitsPerIter = codeLength; % = bitsPerFrame when uncoded
end
totalComparedBits = numComparedBitsPerIter * actual_iterations;

if totalComparedBits == 0
    error("Zero iterations completed or zero bits compared.");
end

if MMSE_flag == 1
    berSC_MMSE   = numErrorsSCinTotal_MMSE   .'/ totalComparedBits;
    berOFDM_MMSE = numErrorsOFDMinTotal_MMSE .'/ totalComparedBits;
    berOCDM_MMSE = numErrorsOCDMinTotal_MMSE .'/ totalComparedBits;
    berAFDM_MMSE = numErrorsAFDMinTotal_MMSE .'/ totalComparedBits;
    berOTFS_MMSE = numErrorsOTFSinTotal_MMSE .'/ totalComparedBits;
    % Handle potential NaNs from parfor errors if any occurred
    berSC_MMSE(isnan(berSC_MMSE)) = 1; % Or some other indicator
    berOFDM_MMSE(isnan(berOFDM_MMSE)) = 1;
    % ... etc for other BERs
end

fprintf('Simulation finished. Total iterations: %d. Total compared bits: %d.\n', actual_iterations, totalComparedBits);


% --- Plotting Results ---
figure;
lineWidth = 1.5;
markerSize = 7;

if MMSE_flag == 1
    semilogy(EBN0, berSC_MMSE, '--o', 'DisplayName', 'SC-MMSE', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
    hold on;
    semilogy(EBN0, berOFDM_MMSE, '-.v', 'DisplayName', 'OFDM-MMSE', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
    semilogy(EBN0, berOCDM_MMSE, ':x', 'DisplayName', 'OCDM-MMSE', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
    semilogy(EBN0, berAFDM_MMSE, '--s', 'DisplayName', 'AFDM-MMSE', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
    semilogy(EBN0, berOTFS_MMSE, '-.*', 'DisplayName', 'OTFS-MMSE', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
    hold off;

    grid on;
    legend('Location', 'southwest');

    % Update title based on LDPC flag
    if LDPC_flag == 1
         titleStr = sprintf('BER Performance with LDPC (K=%d, N=%d, R=%.3f), Mod: %d-PSK, SubBlock=%d', ...
                       infoLength, codeLength, actualRate, mod_order, subBlockLength);
    else
         titleStr = sprintf('BER Performance (Uncoded), Mod: %d-PSK, SubBlock=%d, FrameBits=%d', ...
                       mod_order, subBlockLength, bitsPerFrame);
    end
    title(titleStr);
    xlabel('$E_b/N_0$ (dB)','Interpreter', 'latex');
    ylabel('Bit Error Rate (BER)','Interpreter', 'latex');
    ylim([max(min(berSC_MMSE(berSC_MMSE>0))/10, 1e-7) 1]); % Adjust Y-axis limits dynamically or keep fixed [1e-7 1]
    ax = gca;
    ax.FontSize = 11;
else
    warning("MMSE flag is off. No results to plot.");
    hold off;
end


% --- Cleanup ---
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
    disp('Parallel pool closed.');
end
disp('Simulation script finished.');


% --- Helper Functions ---

% 1. Function to construct Heff from physical parameters (Keep as is)
function Heff_sub = constructHeffFromParams(N_sub, pathGains, delayTapsIdx, dopplerTapsHz, Ts)
% ... (previous implementation) ...
    Heff_sub = zeros(N_sub, N_sub);
    numPaths = length(pathGains);
    k_doppler = dopplerTapsHz * Ts * N_sub; % Normalized Doppler k = nu * Ts * N = nu / (1/(N*Ts)) = nu / doppler_resolution
    time_indices = (0:N_sub-1)';
    for p = 1:numPaths
        h_p = pathGains(p);
        l_p = delayTapsIdx(p); % Discrete delay index
        nu_p = dopplerTapsHz(p); % Doppler shift (Hz)

        % Time domain impulse response component for path p at sample n due to input at n-l_p
        % h_p(n, m) = h_p * delta(n - (m+l_p)) * exp(j*2*pi*nu_p*n*Ts)  where m is input time index
        % Heff(n,m) corresponds to output n, input m
        % Equivalent to: Heff = sum_p h_p * Omega_p * Delta_p
        % Omega_p = diag(exp(j*2*pi*nu_p*Ts*(0:N_sub-1)))
        % Delta_p = circshift(eye(N_sub), l_p, 1) <-- shift rows down (delay output)
        % OR Delta_p = circshift(eye(N_sub), l_p, 2) <-- shift columns right (delay input) - let's use this one consistent with original code

        % Doppler phase matrix (applied to input signal at time m) - Check model carefully!
        % If Doppler applies to the signal *at time n*, it should be diag(exp(j*2*pi*nu_p*Ts*(0:N_sub-1)))
        % Let's stick to the original code's likely interpretation:
        dopplerPhaseRamp = exp(1j * 2 * pi * nu_p * Ts * time_indices); % Phase rotation at each sample time n
        Omega_nu = diag(dopplerPhaseRamp);

        % Circular shift matrix for delay l_p (input at m appears at output m+l_p, wrapped around)
        % circshift(eye(N), l, 2) shifts columns right by l. Input m goes to output n=m. Effect is on input index.
        % Input s[m] contributes to output y[n] via path p as h_p * exp(j*2*pi*nu_p*n*Ts) * s[ (n-l_p)_N ]
        % Heff(n,m) = sum_p h_p * exp(j*2*pi*nu_p*n*Ts) * delta( m - (n-l_p)_N )
        % This looks like Omega_nu * circshift(eye(N_sub), l_p, 1) -- Row shift version
        % Let's re-verify the column shift:
        % y = Hx => y[n] = sum_m H(n,m) * x[m]
        % If H = Omega * Delta_col_shift, y[n] = sum_m Omega(n,n) * Delta_col_shift(n,m) * x[m]
        % Delta_col_shift(n,m) = 1 if m = (n-l_p)_N.
        % y[n] = Omega(n,n) * x[(n-l_p)_N] = exp(j*2*pi*nu_p*n*Ts) * x[(n-l_p)_N]
        % This seems correct for the LTV model y[n] = sum_p h_p * exp(j*2*pi*nu_p*n*Ts) * x[(n-l_p)_N]

        Delta_l_cols = circshift(eye(N_sub), l_p, 2); % Shift columns right

        Heff_sub = Heff_sub + h_p * Omega_nu * Delta_l_cols; % Consistent with original code structure
    end
end

% 2. Function to evolve channel parameters (Keep placeholder or improve)
function [nextGains, nextDelays, nextDopplersHz] = evolveChannelParams(currentGains, currentDelays, currentDopplersHz, blockDuration)
% ... (previous placeholder implementation or replace with a better one) ...
    numPaths = length(currentGains);
    % Example: Basic phase drift based on Doppler (more realistic than pure random phase)
    avgDoppler = mean(abs(currentDopplersHz)); % Crude way to get a sense of speed
    phase_stddev = min(2*pi*avgDoppler*blockDuration, pi); % Limit phase change, related to fD*T
    if isnan(phase_stddev); phase_stddev = 0.05; end % Handle case of zero Doppler

    phase_drift = exp(1j * phase_stddev * randn(numPaths, 1)); % Simplified phase rotation
    nextGains = currentGains .* phase_drift;

    % Delays usually change slower, keep constant for simplicity here
    nextDelays = currentDelays;

    % Doppler might drift slightly
    doppler_drift_stddev_hz = 1; % Hz per block (adjust based on expected oscillator drift / path changes)
    nextDopplersHz = currentDopplersHz + doppler_drift_stddev_hz * randn(numPaths, 1);

    % Optional: Renormalize power if drift causes significant changes
    % currentPower = sum(abs(currentGains).^2);
    % nextPower = sum(abs(nextGains).^2);
    % if nextPower > 1e-9
    %     nextGains = nextGains * sqrt(currentPower / nextPower);
    % end
end

% 3. Function createMatrix (Ensure OFDM output assignment is correct)
function [Tmatrix, ITmatrix] = createMatrix(dataLength, delayMax, dopplerMax, matrixType)
     switch matrixType
         case 'OFDM'
             DFT = (1/sqrt(dataLength))*dftmtx(dataLength);
             IDFT = DFT'; % Or equivalently conj(DFT) because dftmtx is symmetric
             % Assign correctly based on main script usage:
             % createMatrix returns [DFT_sub, IDFT_sub]
             % Tx uses IDFT_sub, Rx uses DFT_sub
             Tmatrix = DFT;  % First output should be DFT
             ITmatrix = IDFT; % Second output should be IDFT
         case 'OCDM'
             % Assuming DFnT verified as unitary
             DFnT = zeros(dataLength, dataLength);
             for i = 1: dataLength
                 m = i - 1;
                 for j = 1: dataLength
                     n = j - 1;
                     DFnT(i, j) = 1/sqrt(dataLength) * exp(1i*pi/dataLength*(m+n)^2);
                 end
             end
             IDFnT = DFnT'; % Correct inverse if DFnT is unitary
             Tmatrix = DFnT;
             ITmatrix = IDFnT;
         case 'AFDM'
             % Assuming DAFT verified as unitary
             DAFT = zeros(dataLength, dataLength);
             c1 = round(2*dopplerMax+1)/(2*dataLength); % Ensure dopplerMax is integer index
             c2 = 1/(2*dataLength);
             for i = 1: dataLength
                 m = i - 1;
                 for j = 1: dataLength
                     n = j - 1;
                     DAFT(i, j) = 1/sqrt(dataLength) * exp(1i*2*pi * (c1*n^2 + c2*m^2 + m*n/dataLength));
                 end
             end
             IDAFT = DAFT'; % Correct inverse if DAFT is unitary
             Tmatrix = DAFT;
             ITmatrix = IDAFT;
         case 'OTFS'
             % Requires dataLength to be a perfect square N=M*M
             symbolsPerFrameForOTFS = sqrt(dataLength);
             sampledPerSymbolForOTFS = symbolsPerFrameForOTFS; % N=M for OTFS grid
             if mod(symbolsPerFrameForOTFS, 1) ~= 0
                 error('OTFS requires dataLength (%d) to be a perfect square.', dataLength);
             end
             DFTforOTFS = (1/sqrt(symbolsPerFrameForOTFS))*dftmtx(symbolsPerFrameForOTFS);
             IDFTforOTFS = DFTforOTFS';
             Window = eye(sampledPerSymbolForOTFS); % Rectangular window
             ISFFT = kron(IDFTforOTFS, Window); % Maps DD (vectorized) to Time domain
             SFFT = ISFFT'; % Maps Time domain back to DD (vectorized). Correct since ISFFT is unitary.
             Tmatrix = SFFT;  % Forward transform (Time -> DD)
             ITmatrix = ISFFT; % Inverse transform (DD -> Time)
     end
end