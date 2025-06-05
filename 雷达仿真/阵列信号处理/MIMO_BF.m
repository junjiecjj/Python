
clear all; close all; clc;
% https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247504602&idx=3&sn=0bbd408624d4a40cc2d5620dea395e6c&chksm=c26724ece8e9066d5f019a4f67a0b4bb37f39b85628ea384ae79d0cfda90acff83e66a77e8e5&mpshare=1&scene=1&srcid=0604jiXbcJNp3SD6xuooZb0A&sharer_shareinfo=97ed50bb3cf25d6d2e1bc30e78162774&sharer_shareinfo_first=97ed50bb3cf25d6d2e1bc30e78162774&exportkey=n_ChQIAhIQDM%2BFub8MMZH%2Fu2bzhJIt4xKfAgIE97dBBAEAAAAAABREMoIZ%2BQoAAAAOpnltbLcz9gKNyK89dVj0YoivGUKc5XxbwK2o%2BRT0qlX%2FEN4q8hCDf83%2BVtJjO2sNU9vQCAnsKehlgKlwK6qU9ZCFvlHH5vBqk56QPPKWYMGMKyBU9KHGls39tndr0FZUj3JRgKg7taEp5uTAdJU1y5xeFl0dZHPQVt%2FZrKYnpPiNKlxaPrBlzovMAnnxipIf7plrYpkKweh1yBQ87XiNQhk5sidQxQ%2BGwzRMUbBX5%2BpMzBFOjKxsF0MsLa%2FkIH6l3510d1DIrhX6ajnoFjffsuDv%2BjZs6O%2FQvfYfQn12Un4KGcWl5QbbQZPzD5xPrFLN6j0o1Mrd15PBD2ElrziuEUD2DguFRATl&acctmode=0&pass_ticket=SVI22YdMOfTX5uPwSMGb9MflpDte2DTqI3s5xPMRV3eUvgZ89twln3MnOpqIgVWS&wx_header=0&poc_token=HIdTQWij3Qp8NY9hXT1fMcLwIzeQWoKdZuRdcz3N

%% Step 1: Set Up the Environment
rng(42); % Set random seed for reproducibility

%% Step 2: Define Antenna Layout
numAntennas = 8;
bsPosition = [0, 0];
lambda = 1;
d = lambda / 2;
antennaPositions = (0:numAntennas-1)' * d;

%% Step 3: Define Devices with Minimum Separation
areaSize = 100;
numDevices = 25;
minSeparation = 10; % Minimum distance between devices
devicePositions = zeros(numDevices, 2);

% Generate device positions with minimum separation
for i = 1:numDevices
    attempts = 0;
    while attempts < 100
        pos = areaSize * (rand(1, 2) - 0.5);
        if i == 1
            devicePositions(i, :) = pos;
            break;
        end
        % Check distance to all previous devices
        distances = sqrt(sum((devicePositions(1:i-1, :) - pos).^2, 2));
        if all(distances >= minSeparation)
            devicePositions(i, :) = pos;
            break;
        end
        attempts = attempts + 1;
    end
    if attempts >= 100
        error('Could not place device %d with minimum separation of %d. Try reducing minSeparation or areaSize.', i, minSeparation);
    end
end

%% Step 4: User Selection for Multiple Communications
numCommunications = input('Enter number of communications (1 to 10): ');
while numCommunications < 1 || numCommunications > 10 || floor(numCommunications) ~= numCommunications
    disp('Invalid input. Try again.');
    numCommunications = input('Enter number of communications (1 to 10): ');
end

commDevices = cell(numCommunications, 1);
% Define colors for up to 10 communications
colors = {'r', 'g', 'b', 'c', 'm', 'y', 'k', [0.5 0 0], [0 0.5 0], [0 0 0.5]};
for comm = 1:numCommunications
    fprintf('Communication %d:\n', comm);
    numDevInComm = input(sprintf('Enter number of devices for communication %d (2 to %d): ', comm, min(numAntennas-1, numDevices)));
    while numDevInComm < 2 || numDevInComm > min(numAntennas-1, numDevices) || floor(numDevInComm) ~= numDevInComm
        disp('Invalid input. Must be between 2 and %d.', min(numAntennas-1, numDevices));
        numDevInComm = input(sprintf('Enter number of devices for communication %d (2 to %d): ', comm, min(numAntennas-1, numDevices)));
    end

    devices = zeros(numDevInComm, 1);
    for i = 1:numDevInComm
        prompt = sprintf('Enter device number %d for communication %d (1 to %d): ', i, comm, numDevices);
        dev = input(prompt);
        % Flatten all previously selected devices into a single array for ismember check
        selectedDevices = [];
        for j = 1:comm-1
            selectedDevices = [selectedDevices; commDevices{j}];
        end
        if comm > 1
            selectedDevices = [selectedDevices; devices(1:i-1)];
        else
            selectedDevices = devices(1:i-1);
        end
        while dev < 1 || dev > numDevices || floor(dev) ~= dev || ismember(dev, selectedDevices)
            disp('Invalid or already selected device. Try again.');
            dev = input(prompt);
        end
        devices(i) = dev;
    end
    commDevices{comm} = devices;
end

%% Step 5: Simulation Setup
SNR = 30; % Increased SNR
noiseVar = 10^(-SNR/10);
numIterations = 5;
receivedPowerMIMO = cell(numCommunications, 1);
receivedPowerSISO = cell(numCommunications, 1);
sinrValues = cell(numCommunications, 1);
for comm = 1:numCommunications
    numStreams = length(commDevices{comm});
    receivedPowerMIMO{comm} = zeros(numStreams, numIterations);
    receivedPowerSISO{comm} = zeros(numStreams, numIterations);
    sinrValues{comm} = zeros(numStreams, numIterations);
end

% Path Loss Model Parameters
PL0 = 30; % Path loss at reference distance d0 (in dB)
d0 = 1; % Reference distance (in meters)
pathLossExponent = 3; % Environment dependent
shadowingStdDev = 1; % Further reduced for stability

%% Step 6: Visualization Setup
fig = figure('Name', 'MIMO Beamforming Visualization', 'Position', [100, 100, 1200, 500]);
movementScale = 5;
numAnimationSteps = 20;

for iter = 1:numIterations
    % Device movement (random walk with minimum separation)
    for i = 1:numDevices
        attempts = 0;
        while attempts < 100
            newPos = devicePositions(i, :) + movementScale * (rand(1, 2) - 0.5);
            newPos = min(max(newPos, -areaSize/2), areaSize/2);
            % Check distance to all other devices
            otherPositions = devicePositions(setdiff(1:numDevices, i), :);
            distances = sqrt(sum((otherPositions - newPos).^2, 2));
            if all(distances >= minSeparation)
                devicePositions(i, :) = newPos;
                break;
            end
            attempts = attempts + 1;
        end
        if attempts >= 100
            warning('Could not move device %d while maintaining minimum separation.', i);
        end
    end

    % Left Subplot: Device Layout and Data Flow
    subplot(1, 2, 1); cla; hold on; grid on; axis equal;
    xlim([-areaSize/2, areaSize/2]); ylim([-areaSize/2, areaSize/2]);
    xlabel('X Position'); ylabel('Y Position');
    title(sprintf('Iteration %d/%d - Device Layout', iter, numIterations));
    plot(bsPosition(1), bsPosition(2), 'ks', 'MarkerSize', 10, 'LineWidth', 2);
    plot(devicePositions(:, 1), devicePositions(:, 2), 'bo', 'MarkerSize', 8);

    % Process each communication
    totalWeights = zeros(numAntennas, 0);
    senderIndices = zeros(numCommunications, 1);
    for comm = 1:numCommunications
        selectedDevices = commDevices{comm};
        numStreams = length(selectedDevices);
        angles = zeros(1, numStreams);
        weights = zeros(numAntennas, numStreams);
        pathGains = zeros(numStreams, 1);

        % Compute angles and path gains (for channel scaling)
        for i = 1:numStreams
            devPos = devicePositions(selectedDevices(i), :) - bsPosition;
            angles(i) = atan2d(devPos(2), devPos(1));
            dist = norm(devPos);
            PL_dB = PL0 + 10 * pathLossExponent * log10(dist / d0) + randn * shadowingStdDev;
            pathGains(i) = 10^(-PL_dB / 10);
        end

        % MIMO channel with path loss scaling (numStreams x numAntennas)
        condThreshold = 50; % Lowered threshold for better conditioning
        H = (randn(numStreams, numAntennas) + 1j * randn(numStreams, numAntennas)) / sqrt(2);
        for i = 1:numStreams
            H(i, :) = sqrt(pathGains(i)) * H(i, :);
        end
        % Regenerate H if condition number is too high
        attempts = 0;
        maxAttempts = 20; % Increased attempts
        while cond(H) > condThreshold && attempts < maxAttempts
            H = (randn(numStreams, numAntennas) + 1j * randn(numStreams, numAntennas)) / sqrt(2);
            for i = 1:numStreams
                H(i, :) = sqrt(pathGains(i)) * H(i, :);
            end
            attempts = attempts + 1;
        end
        if attempts >= maxAttempts
            warning('Could not generate well-conditioned channel matrix for Comm %d after %d attempts.', comm, maxAttempts);
        end

        % MMSE Beamforming (precoding matrix)
        W = pinv(H' * H + noiseVar * eye(numAntennas)) * H';
        % Normalize columns of W to ensure fair power allocation
        for i = 1:numStreams
            W(:, i) = W(:, i) / norm(W(:, i));
        end
        weights = W; % numAntennas x numStreams

        % Symbol vector (e.g., all ones for simplicity)
        s = ones(numStreams, 1); % numStreams x 1
        % Transmit signal: x = W * s
        x = weights * s; % numAntennas x 1
        % Received signal: y = H * x + noise
        receivedSignal = H * x + sqrt(noiseVar) * (randn(numStreams, 1) + 1j * randn(numStreams, 1)); % numStreams x 1

        % Power and SINR
        for i = 1:numStreams
            signalPower = abs(receivedSignal(i))^2; % Scalar power for each stream
            interference = sum(abs(receivedSignal).^2) - signalPower; % Total interference
            sinrValues{comm}(i, iter) = 10 * log10(signalPower / (interference + noiseVar));
            receivedPowerMIMO{comm}(i, iter) = signalPower;
        end

        % SISO baseline with path loss
        for i = 1:numStreams
            hSISO = (randn + 1j * randn) / sqrt(2);
            hSISO = sqrt(pathGains(i)) * hSISO;
            sisoSignal = hSISO * 1 + sqrt(noiseVar) * (randn + 1j * randn); % Single symbol
            receivedPowerSISO{comm}(i, iter) = abs(sisoSignal)^2;
        end

        % Plot selected devices with communication-specific color
        plot(devicePositions(selectedDevices, 1), devicePositions(selectedDevices, 2), ...
            [colors{comm} '*'], 'MarkerSize', 12, 'LineWidth', 2);

        % Select random sender for this communication
        senderIndices(comm) = randi(numStreams);

        % Accumulate weights for beam pattern visualization
        totalWeights = [totalWeights, weights];
    end

    % Simultaneous visualization for all communications
    % Phase 1: Animate all sender-to-BS paths
    flows = cell(numCommunications, 1);
    for comm = 1:numCommunications
        selectedDevices = commDevices{comm};
        senderIdx = senderIndices(comm);
        senderPos = devicePositions(selectedDevices(senderIdx), :);
        plot([senderPos(1), bsPosition(1)], [senderPos(2), bsPosition(2)], ...
            [colors{comm} '--'], 'LineWidth', 1.5);
        flows{comm} = plot(senderPos(1), senderPos(2), [colors{comm} '^'], 'MarkerSize', 8);
    end
    for step = 1:numAnimationSteps
        t = step / numAnimationSteps;
        for comm = 1:numCommunications
            selectedDevices = commDevices{comm};
            senderIdx = senderIndices(comm);
            senderPos = devicePositions(selectedDevices(senderIdx), :);
            x = senderPos(1) + t * (bsPosition(1) - senderPos(1));
            y = senderPos(2) + t * (bsPosition(2) - senderPos(2));
            set(flows{comm}, 'XData', x, 'YData', y);
        end
        drawnow;
        pause(0.01);
    end

    % Phase 2: Animate BS-to-all-receivers paths for all communications
    flows = cell(numCommunications, 1);
    for comm = 1:numCommunications
        selectedDevices = commDevices{comm};
        senderIdx = senderIndices(comm);
        flows{comm} = gobjects(length(selectedDevices)-1, 1);
        receiverCount = 1;
        for i = 1:length(selectedDevices)
            if i ~= senderIdx
                receiverPos = devicePositions(selectedDevices(i), :);
                plot([bsPosition(1), receiverPos(1)], [bsPosition(2), receiverPos(2)], ...
                    [colors{comm} '--'], 'LineWidth', 1.5);
                flows{comm}(receiverCount) = plot(bsPosition(1), bsPosition(2), ...
                    [colors{comm} '^'], 'MarkerSize', 8);
                receiverCount = receiverCount + 1;
            end
        end
    end
    for step = 1:numAnimationSteps
        t = step / numAnimationSteps;
        for comm = 1:numCommunications
            selectedDevices = commDevices{comm};
            senderIdx = senderIndices(comm);
            receiverCount = 1;
            for i = 1:length(selectedDevices)
                if i ~= senderIdx
                    receiverPos = devicePositions(selectedDevices(i), :);
                    x = bsPosition(1) + t * (receiverPos(1) - bsPosition(1));
                    y = bsPosition(2) + t * (receiverPos(2) - bsPosition(2));
                    set(flows{comm}(receiverCount), 'XData', x, 'YData', y);
                    receiverCount = receiverCount + 1;
                end
            end
        end
        drawnow;
        pause(0.01);
    end

    % Right Subplot: Radiation Pattern
    subplot(1, 2, 2); cla;
    thetaSweep = linspace(-pi, pi, 360);
    beamPattern = zeros(1, length(thetaSweep));
    totalWeight = sum(totalWeights, 2);
    for t = 1:length(thetaSweep)
        sv = exp(-1j * 2 * pi * d * (0:numAntennas-1)' * sin(thetaSweep(t)) / lambda);
        beamPattern(t) = abs(sv' * totalWeight);
    end
    beamPattern = 20 * log10(beamPattern / max(beamPattern));
    beamPattern(beamPattern < -30) = -30;
    polarplot(thetaSweep, beamPattern, 'r', 'LineWidth', 1.5);
    title('Beam Pattern of Antenna Array');
    rlim([-30 0]);

    pause(0.2);
end

%% Final Plots and Metrics
figure('Name', 'MIMO + Beamforming vs SISO by Communication');
hold on;
barWidth = 0.4;
avgPowerMIMOComm = zeros(numCommunications, 1);
avgPowerSISOComm = zeros(numCommunications, 1);

for comm = 1:numCommunications
    selectedDevices = commDevices{comm};
    avgPowerMIMO = mean(receivedPowerMIMO{comm}, 2);
    avgPowerSISO = mean(receivedPowerSISO{comm}, 2);
    avgPowerMIMOComm(comm) = mean(avgPowerMIMO);
    avgPowerSISOComm(comm) = mean(avgPowerSISO);
end

x = 1:numCommunications;
bar(x - barWidth/2, 10*log10(avgPowerMIMOComm), barWidth, 'FaceColor', 'r', 'DisplayName', 'MIMO + Beamforming');
bar(x + barWidth/2, 10*log10(avgPowerSISOComm), barWidth, 'FaceColor', [0.5 0.5 0.5], 'DisplayName', 'SISO');

% Determine which system performs better and add annotations
for comm = 1:numCommunications
    mimoPower = 10*log10(avgPowerMIMOComm(comm));
    sisoPower = 10*log10(avgPowerSISOComm(comm));
    if mimoPower > sisoPower
        betterSystem = 'MIMO better';
        yPos = mimoPower + 1; % Position text above the higher bar
    else
        betterSystem = 'SISO better';
        yPos = sisoPower + 1;
    end
    text(comm, yPos, betterSystem, 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'k');
end

xlabel('Communication');
ylabel('Avg Received Power (dB)');
title('MIMO + Beamforming vs SISO by Communication');
legend('Location', 'best');
xticks(1:numCommunications);
xticklabels(arrayfun(@(x) sprintf('Comm %d', x), 1:numCommunications, 'UniformOutput', false));
grid on;

% Console Output
disp('Communication Metrics:');
fprintf('SNR: %.2f dB\n', SNR);
for comm = 1:numCommunications
    selectedDevices = commDevices{comm};
    avgSINR = mean(sinrValues{comm}, 2);
    fprintf('Metrics for Communication %d:\n', comm);
    for i = 1:length(selectedDevices)
        fprintf('Device %d - Final MIMO Power: %.2f dB, Avg SINR: %.2f dB\n', ...
            selectedDevices(i), 10*log10(receivedPowerMIMO{comm}(i, end)), avgSINR(i));
        fprintf('          Final SISO Power: %.2f dB\n', 10*log10(receivedPowerSISO{comm}(i, end)));
    end
end

% Print performance comparison for each communication
disp('Performance Comparison:');
for comm = 1:numCommunications
    mimoPower = 10*log10(avgPowerMIMOComm(comm));
    sisoPower = 10*log10(avgPowerSISOComm(comm));
    if mimoPower > sisoPower
        fprintf('Communication %d: MIMO performs better with avg power %.2f dB vs SISO %.2f dB\n', ...
            comm, mimoPower, sisoPower);
    else
        fprintf('Communication %d: SISO performs better with avg power %.2f dB vs MIMO %.2f dB\n', ...
            comm, sisoPower, mimoPower);
    end
end