%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copywrite Mohammed Abuibaid, m.a.abuibaid@gmail.com, 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adaptive BeamForming Constant Modulus CM Algorithm
clear all; close all; clc; 
format long;set(0,'defaultaxesfontsize',20);

%% Simulation Parameters
% MSK ModDem Configuration
ini_phase = pi/2;
SamplesPerSymbol = 2;
dataenc = 'nondiff';
%rng('shuffle')

bit_count = 1e4;
SIR1 = 10; %dB, Signal-to-Interference Ratio with Interferer 1
SIR2 = 10; %dB, Signal-to-Interference Ratio with Interferer 2
SNR = -12:2:15; %Range of SNR

uncoded_bits_Tx = randi([0 1],bit_count,1);
uncoded_bits1 = randi([0 1],bit_count,1);
uncoded_bits2 = randi([0 1],bit_count,1);


% Uniform Linear Array (ULA) Antennas' Configuration
NumofAntenna = 10; % Number of Antennas in Uniform Linear Array (ULA)
Kd = pi; % Assuming Antennas are seperated by lambda/2
theta_tx = (60)*(pi/180); % Direction of Arraival(DoA)of Tx
theta_Int1 = (30)*(pi/180); % Direction of Arraival(DoA)of Interferer 1
theta_Int2 =  (120)*(pi/180); % Direction of Arraival(DoA)of Interferer 2

% Plotting Configuration
plott = true;
plott_EyeDiagram = true;
BER = NaN*ones(1,length(SNR));
errorStats = zeros(3,length(SNR));
NumofErrors = zeros(1,length(SNR));


%% Specifying the range of SNR to carry out simulation
for SNRi = length(SNR)
%for SNRi = 1:length(SNR)

    disp(['Current SNR = ' num2str(SNR(SNRi)) '  Last SNR = ' num2str(SNR(end)) ])
    T_Errors = 0;
    T_bits = 0;
    
    while T_Errors < 1  % Min errors = 1  through 10000 bits
        
        %Create an MSK modulator, an AWGN channel, and an MSK demodulator.
        %Create an error rate calculator, account for the delay caused by the Viterbi algorithm
        hMod = comm.MSKModulator('BitInput', true,'InitialPhaseOffset', ini_phase,'SamplesPerSymbol',SamplesPerSymbol);
        hAWGN = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR',SNR(SNRi));
        hDemod = comm.MSKDemodulator('BitOutput', true,'InitialPhaseOffset', ini_phase,'SamplesPerSymbol',SamplesPerSymbol);
        hError = comm.ErrorRate('ReceiveDelay', hDemod.TracebackDepth);
        
        %Tx: Generation amd Modulation Information Bits
        MSK_TxSig = step(hMod, uncoded_bits_Tx);
        
        %Interferer 1: Generation amd Modulation Information Bits
        Inter1 = 1/10^(SIR1/10);
        MSK_Int1 = sqrt(Inter1/2)* step(hMod, uncoded_bits1);
        
        %Interferer 2: Generation amd Modulation Information Bits
        Inter2 = 1/10^(SIR2/10);
        MSK_Int2 = sqrt(Inter2/2)*step(hMod, uncoded_bits2);
        
        % Array Propagation Vectors
        APV_TxSig = exp(1j*Kd*(0:NumofAntenna-1)'*cos(theta_tx));
        APV_Int1 = exp(1j*Kd*(0:NumofAntenna-1)'*cos(theta_Int1));
        APV_Int2 = exp(1j*Kd*(0:NumofAntenna-1)'*cos(theta_Int2));
        
        Tx_Sig =APV_TxSig*MSK_TxSig.';
        Int1 = APV_Int1*MSK_Int1.';
        Int2 = APV_Int2*MSK_Int2.';
        
        Rx_Sig_Int = Tx_Sig + Int1 + Int2;
        Rx_Sig = step(hAWGN, Rx_Sig_Int);
        
        % Constant Modulus�(CM) Algorithm
        %start weight vector, may be in the direction of the desired signal:
        w_time = zeros(NumofAntenna,length(MSK_TxSig));
        w = 0.25*APV_TxSig.';
        %start vector may be any type:
        %w = [0.1 0.1 0.1 0.1]
        mu = 0.01;
        y = zeros(1,length(MSK_TxSig));
        e = zeros(1,length(MSK_TxSig));
                
        %Constant Modulus Algorithm Computation
        for n=1:length(MSK_TxSig),
            y(n) = w*Rx_Sig(:,n);
            e(n) = y(n)/((abs(y(n))))^2-y(n); %SATO's principle for updating the error
            w = w + mu*e(n)*Rx_Sig(:,n)';
            w_time(:,n) = w.';
        end
        
        receivedData = step(hDemod,y.');
        errorStats(:,SNRi) = step(hError, uncoded_bits_Tx, receivedData);
        
        T_Errors = T_Errors + errorStats(2,SNRi);
        T_bits = T_bits + length(uncoded_bits_Tx);
        disp(['T_Errors = ' num2str(T_Errors)])
        
    end %while T_Errors < 3
    
    %Calculate Number of Errors
    NumofErrors(SNRi) = T_Errors;
    % Calculate Bit Error Rate
    BER(SNRi) = T_Errors / T_bits;
    disp(['BER = ' num2str(BER(SNRi),'%.e')])
    
    if plott_EyeDiagram == true
        %%
        set(0,'defaultaxesfontsize',15);
        period = 1; offset = 0;
        % Trasmitted Signal
        eyediagram(MSK_TxSig,SamplesPerSymbol,period,offset,'-b')
        % Received Signal withOUT BF
        WithOutBF  = sum(Rx_Sig);
        eyediagram(WithOutBF,SamplesPerSymbol,period,offset,'-b')
        % Received Signal with BF
        eyediagram(y,SamplesPerSymbol,period,offset,'-b')
    end
    
end


if plott == true
    
    % Rx Array Antenna Response
    limR = 1; % Span of radius in Radiation Pattern
    theta = linspace(0,180,400);
    ArrayFactor = exp(1j*Kd*(0:NumofAntenna-1)'*cos(deg2rad(theta)));
    Rad_Pattern = abs(w*ArrayFactor);
    
    % X-Y Plotting
    figure('name','Rx Array X-Y Response Pattern')
    plot(theta, 20*log10(Rad_Pattern),'LineWidth', 1.2);
    grid on; hold on;
    XYLimits = axis;
    line('XData', (180/pi)*[theta_tx theta_tx], 'YData', [XYLimits(3) XYLimits(4)], 'LineStyle', '--','LineWidth', 1.2, 'Color','g');
    line('XData', (180/pi)*[theta_Int1 theta_Int1], 'YData', [XYLimits(3) XYLimits(4)], 'LineStyle', '--','LineWidth', 1.2, 'Color','r');
    line('XData', (180/pi)*[theta_Int2 theta_Int2], 'YData', [XYLimits(3) XYLimits(4)], 'LineStyle', '--','LineWidth', 1.2, 'Color','r');
    title('Rx ULA Gain');
    ylabel('Gain (dB)');
    xlabel('Angle (Degree)');
    xlim(180*[0 1])
    ylim([-55 10])
    legend('Rx Rad. Pattern', ...
        ['Target Loc @ ' num2str(theta_tx *180/pi)], ...
        ['Interferer1 @  ' num2str(theta_Int1 *180/pi)], ...
        ['Interferer2 @ ' num2str(theta_Int2 *180/pi)], ...
        'location', 'best');
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
    % set(gca,'fontsize',20)
    
    
    % Polar Radiation Pattern Plotting
    % set(0,'defaultaxesfontsize',20);
    figure('name','Rx Array Polar Response Pattern')
    TxArray = polar(deg2rad(theta),Rad_Pattern,'-b');
    set(TxArray, 'LineWidth',1)
    grid on; hold on;
    Tx_Loc = polar(theta_tx, limR, 'ok');
    set(Tx_Loc, 'MarkerFaceColor', 'g')
    InterfererLoc1 = polar(theta_Int1, limR, '*r');
    set(InterfererLoc1, 'MarkerFaceColor', 'r')
    InterfererLoc2 = polar(theta_Int2, limR, '*r');
    set(InterfererLoc2, 'MarkerFaceColor', 'r')
%     legend([TxArray,Tx_Loc,InterfererLoc1,InterfererLoc2],...
%         'Rx Rad. Pattern', ...
%         ['Target Loc @ ' num2str(theta_tx *180/pi)], ...
%         ['Interferer1 @  ' num2str(theta_Int1 *180/pi)], ...
%         ['Interferer2 @ ' num2str(theta_Int2 *180/pi)], ...
%         'Position',[217 130 0.01 0.01 ],'FitBoxToText','on');
    
            
    %% Error and Wieght Vector Response Convergence
    
    NumOfSamples = 1e3; %Span of x-axis in Convergence Plots
    figure('name','Amplitude of Error')
    plot(abs(e),'-b','linewidth',0.5)
    grid on;
    axis([1 NumOfSamples 0 1])
    title('Amplitude of Error')
    ylabel('Error Amplitude')
    % set(gca,'fontsize',20)
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
    
    figure('name','Phase of Error')
    plot(rad2deg(angle(e)),'-r','linewidth',0.5)
    grid on;
    axis([1 NumOfSamples -360 360])
    title('Phase of Error')
    ylabel('Phase (Degree)')
    xlabel('Number of Samples')
    % set(gca,'fontsize',20)
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
    
    %%
    
    NumOfSamples = 2e3; %Span of x-axis in Convergence Plots
    figure('name','W Amplitude Convergence')
    plot(smooth(abs(w_time(1,:))),'-r','linewidth',1)
    hold on; grid on;
    plot(smooth(abs(w_time(2,:))),'-b','linewidth',1)
    plot(smooth(abs(w_time(3,:))),'-k','linewidth',1)
    plot(smooth(abs(w_time(4,:))),'-g','linewidth',1)
    xlim([1 NumOfSamples])
    title('Amplitude of Weight Vector Coefficients')
    ylabel('Coefficient Amplitude')
    legend('w_0', 'w_1', 'w_2', 'w_3','location', 'best')
    % plot(smooth(abs(w_time(5,:))),'-m','linewidth',1)
    % plot(smooth(abs(w_time(6,:))),'-c','linewidth',1)
    % plot(smooth(abs(w_time(7,:))),'-y','linewidth',1)
    % legend('w_0', 'w_1', 'w_2', 'w_3','w_4', 'w_5', 'w_6', 'w_7','location', 'best')
    % set(gca,'fontsize',20)
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
    
    figure('name','W Phase Convergence')
    plot(smooth(rad2deg(angle(w_time(1,:)))),'-r','linewidth',1)
    hold on; grid on;
    plot(smooth(rad2deg(angle(w_time(2,:)))),'-b','linewidth',1)
    plot(smooth(rad2deg(angle(w_time(3,:)))),'-k','linewidth',1)
    plot(smooth(rad2deg(angle(w_time(4,:)))),'-g','linewidth',1)
    axis([1 NumOfSamples -360 360])
    title('Phase of Weight Vector Coefficients')
    ylabel('Phase (Degree)')
    xlabel('Number of Samples')
    legend('w_0', 'w_1', 'w_2', 'w_3','location', 'best')
    % plot(smooth(rad2deg(angle(w_time(5,:)))),'-m','linewidth',1)
    % plot(smooth(rad2deg(angle(w_time(6,:)))),'-c','linewidth',1)
    % plot(smooth(rad2deg(angle(w_time(7,:)))),'-y','linewidth',1)
    % legend('w_0', 'w_1', 'w_2', 'w_3','w_4', 'w_5', 'w_6', 'w_7','location', 'best')
    % set(gca,'fontsize',20)
    % set(gcf,'units','normalized','outerposition',[0 0 1 1])
end
