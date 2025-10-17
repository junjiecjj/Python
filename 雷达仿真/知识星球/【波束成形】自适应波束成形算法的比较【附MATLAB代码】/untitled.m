%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Copywrite Mohammed Abuibaid, m.a.abuibaid@gmail.com, 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adaptive BeamForming Algorithms (LMS, NLMS and RLS)
clear all; close all; clc; 
format long;set(0,'defaultaxesfontsize',20);

%% Simulation Parameters

% Adaptive Algorithm Selection
Algo = 3;
NumofAntenna = 10; % Number of Antennas in Uniform Linear Array (ULA)

% 1 : Least Mean Squares (LMS) Algorithm
% 2 : Normalized LMS Algorithm
% 3 : Recursive Least Square�(RLS) Algorithm

ModuRadians = pi/4; % QPSK Modulation Radians

bit_count = 1e6; % Packet Size (1 Mbit)
SIR1 = 3; %dB, Signal-to-Interference Ratio with Interferer 1
SIR2 = 3; %dB, Signal-to-Interference Ratio with Interferer 2

Eb_No = -12:2:15; %Range of SNR
SNR = Eb_No + 10*log10(2); %Convert Eb/No values to channel SNR

% Uniform Linear Array (ULA) Antennas' Configuration
Kd = pi; % Assuming Antennas are seperated by lambda/2
theta_tx = (60)*(pi/180); % Direction of Arraival(DoA)of Tx
theta_Int1 = (30)*(pi/180); % Direction of Arraival(DoA)of Interferer 1
theta_Int2 =  (120)*(pi/180); % Direction of Arraival(DoA)of Interferer 2

% Plotting Configuration
BER = NaN*ones(1,length(SNR));
NumofErrors = zeros(1,length(SNR));
plott = false;
plot_Constelation = false; % !!! Do NOT set it true when Simulationg so long SNR vector
plottError = true;
plottRP = true;

%% Specifying the range of SNR to carry out simulation
for SNRi = 12
%for SNRi = [5 7];
%for SNRi = 1:length(SNR)
    
    disp(['Current SNR = ' num2str(SNR(SNRi)) '  Last SNR = ' num2str(SNR(end)) ])
    
    % Initiate variables
    T_Errors = 0;
    T_bits = 0;
    
     while T_Errors < 3  % Min errors = 3  through 1e6 bits
        
        %Tx: Generation amd Modulation Information Bits
        uncoded_bits_Tx  = round(rand(1,bit_count));
        % Split the stream into two streams, for Quadrature Carriers
        B1 = uncoded_bits_Tx(1:2:end);
        B2 = uncoded_bits_Tx(2:2:end);
        % QPSK modulator with Gray Code
        QPSK_TxSig = ((B1==0).*(B2==0)*(exp(1i*ModuRadians))+....
            (B1==0).*(B2==1)*(exp(3i*ModuRadians))+....
            (B1==1).*(B2==1)*(exp(5i*ModuRadians))+....
            (B1==1).*(B2==0)*(exp(7i*ModuRadians)));
        
        
        %Interferer 1: Generation amd Modulation Information Bits
        uncoded_bits1  = round(rand(1,bit_count));
        % Split the stream into two streams, for Quadrature Carriers
        B1 = uncoded_bits1(1:2:end);
        B2 = uncoded_bits1(2:2:end);
        % QPSK modulator with Gray Code
        Inter1 = 1/10^(SIR1/10);
        QPSK_Int1  = sqrt(Inter1/2)*((B1==0).*(B2==0)*(exp(1i*ModuRadians))+....
            (B1==0).*(B2==1)*(exp(3i*ModuRadians))+....
            (B1==1).*(B2==1)*(exp(5i*ModuRadians))+....
            (B1==1).*(B2==0)*(exp(7i*ModuRadians)));
        
        % Interferer 2: Generation amd Modulation Information Bits
        uncoded_bits2  = round(rand(1,bit_count));
        %Split the stream into two streams, for Quadrature Carriers
        B1 = uncoded_bits2(1:2:end);
        B2 = uncoded_bits2(2:2:end);
        % QPSK modulator with Gray Code
        Inter2 = 1/10^(SIR2/10);
        QPSK_Int2  = sqrt(Inter2/2)*((B1==0).*(B2==0)*(exp(1i*ModuRadians))+....
            (B1==0).*(B2==1)*(exp(3i*ModuRadians))+....
            (B1==1).*(B2==1)*(exp(5i*ModuRadians))+....
            (B1==1).*(B2==0)*(exp(7i*ModuRadians)));
        
        % Array Propagation Vectors
        APV_TxSig = exp(1j*Kd*(0:NumofAntenna-1)'*cos(theta_tx));
        APV_Int1 = exp(1j*Kd*(0:NumofAntenna-1)'*cos(theta_Int1));
        APV_Int2 = exp(1j*Kd*(0:NumofAntenna-1)'*cos(theta_Int2));
        
        Tx_Sig =APV_TxSig*QPSK_TxSig;
        Int1 = APV_Int1*QPSK_Int1;
        Int2 = APV_Int2*QPSK_Int2;
        
        % AWGN Noise at Each branch of Rx variances
        N0 = 1/10^(SNR(SNRi)/10);
        NoiseRX = sqrt(N0/2)*(randn(NumofAntenna,length(QPSK_TxSig)) + 1i*randn(NumofAntenna,length(QPSK_TxSig)));
        
        % Recevied Signal that contaminated with AWGN and Interfernce
        Rx_Sig = (Tx_Sig+ Int1 + Int2 + NoiseRX); % total received signal
        
        
        % Adaptive Beamforming Filter Configuration
        y = zeros(1,length(QPSK_TxSig)); % output
        mu = 0.05; % gradient constant
        e = zeros(1,length(QPSK_TxSig)); % error
        % Adaptive Algorithm: Weight Vector w Calculation
        switch Algo

            case 1 % Least Mean Squares (LMS) Algorithm
                w = zeros(1,NumofAntenna);
                w_time = zeros(NumofAntenna,length(QPSK_TxSig));
                for n=1:length(QPSK_TxSig)
                    y(n) = w * Rx_Sig(:,n);
                    e(n) = QPSK_TxSig(n)-y(n);
                    w = w + mu*e(n)*(Rx_Sig(:,n))';
                    w_time(:,n)=w.';
                end
                
 
            case 2 % Normalized LMS Algorithm
                w = zeros(1,NumofAntenna);
                w_time = zeros(NumofAntenna,length(QPSK_TxSig));
                for n=1:length(QPSK_TxSig)
                    y(n) = w * Rx_Sig(:,n);
                    e(n) = QPSK_TxSig(n)-y(n);
                    w = w + mu*e(n)*(Rx_Sig(:,n))'/(Rx_Sig(:,n)'*Rx_Sig(:,n));
                    w_time(:,n)=w.';
                end
                

            case 3 % Recursive Least Square�(RLS) Algorithm
                w = zeros(NumofAntenna,1);
                lambda= 0.75;
                delta= 1e-2;
                P = 1/delta*eye(NumofAntenna);
                for n=1:length(QPSK_TxSig)
                    y(n) = w.'*Rx_Sig(:,n);
                    e(n) = QPSK_TxSig(n)-y(n);
                    alpha = QPSK_TxSig(:,n)- Rx_Sig(:,n).'*w;
                    g1 = (P*conj(Rx_Sig(:,n)));
                    g2 = (lambda+ Rx_Sig(:,n).'*P*conj(Rx_Sig(:,n)));
                    g = g1/g2;
                    P=(1/lambda)*P-g*Rx_Sig(:,n).'*(1/lambda)*P;
                    w = w + alpha*g;
                    w_time(:,n)=w;
                end
                w = w_time(:,end).';
            otherwise
                disp('Unknown Method has been Selected !!')
        end
        
        
        % QPSK demodulator at the Receiver
        QPSK_RxSig = y;
        B4 = (real(QPSK_RxSig)<0);
        B3 = (imag(QPSK_RxSig)<0);
        uncoded_bits_rx = zeros(1,2*length(QPSK_RxSig));
        uncoded_bits_rx(1:2:end) = B3;
        uncoded_bits_rx(2:2:end) = B4;
        
        % Calculate Num Errors
        diff = uncoded_bits_Tx - uncoded_bits_rx;
        T_Errors = T_Errors + sum(abs(diff));
        T_bits = T_bits + length(uncoded_bits_Tx);
        disp(['T_Errors = ' num2str(T_Errors)])
        
     end % while T_Errors < 1
    NumofErrors(SNRi) = T_Errors;
    
    % Calculate Bit Error Rate
    BER(SNRi) = T_Errors/T_bits;
    disp(['BER = ' num2str(BER(SNRi),'%.e')])
    
    
    if plot_Constelation == true
        %% Received Symbols Constellation
        border = 2.5; % Span of axes in Constelation Diagram
        bit_Errors_Index = abs(diff);
        B3_sym_Errors_Index = bit_Errors_Index(1:2:end);
        B4_sym_Errors_Index = bit_Errors_Index(2:2:end);
        sym_Errors_Index = B4_sym_Errors_Index + B3_sym_Errors_Index;
        sym_Errors_Index = logical(sym_Errors_Index);
        sym_Correct_Index = ~sym_Errors_Index;
        figure; clf;
        hold on; grid on; box on;
        plot(real(QPSK_RxSig(sym_Correct_Index)),imag(QPSK_RxSig(sym_Correct_Index)),'.g','MarkerSize',5)
        plot(real(QPSK_RxSig(sym_Errors_Index)),imag(QPSK_RxSig(sym_Errors_Index)),'.r','MarkerSize', 5)
        axis(border*[-1 1 -1 1])
        line('XData', border*[-1 1], 'YData',  [0 0], 'LineStyle', '-','LineWidth', 1, 'Color','b');
        line('XData',   [0 0], 'YData', border*[-1 1], 'LineStyle', '-','LineWidth', 1, 'Color','b');
        title(['Received Symbols Constellation Diagram ( SNR = ', num2str(SNR(SNRi)) ' dB)']);
        xlabel('I Channel'); ylabel('Q Channel');
        annotation(gcf,'textbox',[0.7 0.8 0.1 0.1],'String',['BER = ' num2str(BER(SNRi),'%.e')],'fontsize',20,'FitBoxToText','on');
        %set(gca,'fontsize',20)
        %set(gcf,'units','normalized','outerposition',[0 0 1 1])
        
    end
end

%%
if plottRP == true
%%    
    % Rx Array Antenna Response
    limR = 1; % Span of radius in Radiation Pattern
    theta = linspace(0,180,400);
    ArrayFactor = exp(1j*Kd*(0:NumofAntenna-1)'*cos(deg2rad(theta)));
    Rad_Pattern = abs(w*ArrayFactor);
    
%     % X-Y Radiation Pattern Plotting
%     figure('name','Rx Array X-Y Response Pattern')
%     plot(theta, 20*log10(Rad_Pattern),'LineWidth', 1.2);
%     grid on; hold on;
%     XYLimits = axis;
%     line('XData', (180/pi)*[theta_tx theta_tx], 'YData', [XYLimits(3) XYLimits(4)], 'LineStyle', '--','LineWidth', 1.2, 'Color','g');
%     line('XData', (180/pi)*[theta_Int1 theta_Int1], 'YData', [XYLimits(3) XYLimits(4)], 'LineStyle', '--','LineWidth', 1.2, 'Color','r');
%     line('XData', (180/pi)*[theta_Int2 theta_Int2], 'YData', [XYLimits(3) XYLimits(4)], 'LineStyle', '--','LineWidth', 1.2, 'Color','r');
%     title('Rx ULA Gain');
%     ylabel('Gain (dB)');
%     xlabel('Angle (Degree)');
%     xlim(180*[0 1])
%     ylim([-40 10])
%     legend('Rx Rad. Pattern', ...
%         ['Target Loc @ ' num2str(theta_tx *180/pi)], ...
%         ['Interferer1 @  ' num2str(theta_Int1 *180/pi)], ...
%         ['Interferer2 @ ' num2str(theta_Int2 *180/pi)], ...
%         'location', 'best');
%     % set(gcf,'units','normalized','outerposition',[0 0 1 1])
%     % set(gca,'fontsize',20)
    
    
    %Polar Radiation Pattern Plotting
    figure('name','Rx Array Polar Response Pattern')
    %TxArray = polar2(deg2rad(theta),Rad_Pattern,[0 1.2],'-b');
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
%         'Position',[217 110 0.01 0.01 ],'FitBoxToText','on');
        
 
end 
if plott == true
    %% BER Performance
    figure('name','BER Performance')
    % BER through Simulation
    semilogy(SNR,BER,'o:r','LineWidth',1.4)
    grid on;hold on; box on;
    
    % Rayleigh: Theoretical BER
    Eb_No_temp = -10:2:50; %Range of SNR
    SNR_temp = Eb_No_temp + 10*log10(2); %Convert Eb/No values to channel SNR
    EbN0Lin = 10.^(Eb_No_temp/10);
    theoryBER_Ray = 0.5.*(1-sqrt(EbN0Lin./(EbN0Lin+1)));
    semilogy(SNR_temp,theoryBER_Ray,'-sb','LineWidth',1.4)
    % AWGN: Theoretical BER through
    theoryBER_AWGN = 0.5*erfc(sqrt(10.^(Eb_No_temp/10)));
    semilogy(SNR_temp,theoryBER_AWGN,'-g','LineWidth',1.4)
    
    axis([SNR_temp(1) SNR_temp(end) 2.1e-6 1e0])
    %axis([SNR_temp(1) SNR_temp(end) min(BER) 1e0])
    legend('Adaptive Beamforming', 'Theoretical Rayleigh', 'Theroretical AWGN')
    xlabel('SNR (dB)')
    ylabel('BER')
    title('QPSK Simulation with Adaptive Beamforming')
    %set(gca,'fontsize',20)
    %set(gcf,'units','normalized','outerposition',[0 0 1 1])
    
    %% Adding Comments on BER Performance Graphs
    %     % Create doublearrow
    %     annotation(gcf,'doublearrow',[0.260416666666667 0.259722222222222],...
    %         [0.30779054916986 0.800684727080904]);
    %
    %     % Create doublearrow
    %     annotation(gcf,'doublearrow',[0.239583333333333 0.620138888888889],...
    %         [0.435504469987229 0.435422913543228]);
    %
    %     % Create textbox
    %     annotation(gcf,'textbox',...
    %         [0.262500000000002 0.55978260869565 0.1625 0.0787869953911986],...
    %         'String',{'BER: 146.4e-3 > 0.49e-6','         SNR 3.01 dB'},...
    %         'FontWeight','bold',...
    %         'FontSize',14,...
    %         'FitBoxToText','off',...
    %         'EdgeColor',[1 1 1],...
    %         'BackgroundColor',[1 1 1]);
    %
    %     % Create textbox
    %     annotation(gcf,'textbox',...
    %         [0.317361111111111 0.373120232730805 0.261805555555556 0.0523627063552083],...
    %         'String','Gain = 29.5 dB, BER = 0.396 e-3',...
    %         'FontWeight','bold',...
    %         'FontSize',18,...
    %         'FitBoxToText','off',...
    %         'EdgeColor',[1 1 1],...
    %         'BackgroundColor',[1 1 1]);
    
end 
if plottError == true
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