%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author: Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:   1 September 2023, 21 March 2024
    @Home:   https://github.com/Spratm-Asleaf/Robust-Waveform

    @Acknowledgement: https://mathworks.com/help/phased/ug/waveform-design-for-a-dual-function-mimo-radcom-system.html
    @Dependencies: Phased Array System Toolbox, Communications Toolbox, Signal Processing Toolbox, DSP System Toolbox
%}

Rho = 0:0.1:1;

% Unify to uppercase
PowerType = upper(PowerType);

NumRho = length(Rho);

NumTheta = 50;                                  % How many "theta" to be tested for each episode
Theta = linspace(0, epsilon*4, NumTheta);     % Compared to "epsilon", "theta" should not be overly large

MC = 1000;                                             % How many Monte-Carlo episodes (aka: trials, scenarios) to run

% Assign Storage
XBar = cell(NumRho, 1);                                % Nominall Waveforsm against "rho"
Xr   = cell(NumRho, NumTheta);                         % Robust Waveform against "rho" and "theta"
Hr   = cell(NumRho, NumTheta);                         % Worst-case channel associated with Robust Waveforms Xr

% Achievalbe Sum Rate
NominalSumRate        = zeros(NumRho, 1);                % The estimated sum_rate claimed by nominal solutions
RealSumRateNominal    = zeros(NumRho, MC);               % The real sum_rate evaluated with nominal solutions

RobustSumRate         = zeros(NumRho, NumTheta);         % The estimated sum_rate claimed by robust solutions
RealSumRateRobust     = cell(NumRho, 1);                 % The real sum_rate evaluated with robust solutions

% Symbol Error Rate
RealSymErrRateNominal = zeros(NumRho, MC);
RealSymErrRateRobust  = cell(NumRho, 1);

for rho_indx = 1:NumRho
    rho = Rho(rho_indx);

    disp(['Testing rho = ' num2str(rho)]);

    % Clip rho for numerical stability
    if rho >= 1 - 1e-3
        rho = 1 - 1e-3;
    elseif rho <= 1e-3
        rho = 1e-3;
    end
    
    % Nominal Design
    disp('      Nominal Design:');
    tic;
    switch PowerType
        case 'TPC'
            [~, XBar{rho_indx}] = g1(HBar, S, Xs, Pt, rho);
        case 'PAPC'
            [~, XBar{rho_indx}] = g2(HBar, S, Xs, Pt, rho);
        otherwise
            error('WaveformDesign_Joint :: Error in Power-Constraint Type :: Non-exist!');
    end
    disp('            ----------');
    disp(['            Time: ' num2str(toc) '(s)']);
    disp('            ----------');
    
    % Robust Design; every "theta" gives different result
    TotalRunningTime = 0;
    disp('      Robust Design:');
    for i = 1:NumTheta
        tic;
        [~, Xr{rho_indx, i}, Hr{rho_indx, i}] = GetRobustSolution_Method3(HBar, XBar{rho_indx}, S, Theta(i), Xs, Pt, rho, PowerType);    % Method 3
        TotalRunningTime = TotalRunningTime + toc;
        disp(['            Trying theta = ' num2str(Theta(i)) ' *** Current Progress: ' num2str(100*i/NumTheta) '%']);
    end
    disp('            ----------');
    disp(['            Time (Average): ' num2str(TotalRunningTime/NumTheta) '(s)']);
    disp('            ----------');
    
    %% Main Simulation Loop
    NominalSumRate(rho_indx, 1)     = GetAvrgSumRate(HBar, XBar{rho_indx}, S, N0);
    for i = 1:NumTheta
        RobustSumRate(rho_indx, i)  = GetAvrgSumRate(Hr{rho_indx, i}, Xr{rho_indx, i}, S, N0);
    end
    disp('Monte-Carlo Waveform Testing ... (Stay Patient! This is quick!)');

    RealSumRateRobust_Local     = zeros(MC, NumTheta);
    RealSymErrRateRobust_Local  = zeros(MC, NumTheta);

    for mc = 1:MC
        % Real channel varies from one realization to another
        H0 = HRef + ((2*rand(size(HRef)) - 1) + (2*rand(size(HRef)) - 1) * 1j)*epsilon;
        W = sqrt(N0)*GetChannelNoise(K, L, 'GAUSSIAN');         % Channel noise
        % NB: The actual received signal is Y = H0*X + W, given waveform X
        
        %% Waveform Evaluation
        % Robust Waveform
        for i = 1:NumTheta
            RealSumRateRobust_Local(mc, i)      = GetAvrgSumRate(H0, Xr{rho_indx, i}, S, N0);
    
            rd = pskdemod(H0*Xr{rho_indx, i} + W, Q, pi/Q);                                   
            [~, errRate] = symerr(data, rd);
            RealSymErrRateRobust_Local(mc, i)   = errRate;
        end
        
        % Nominal Waveform
        RealSumRateNominal(rho_indx, mc)      = GetAvrgSumRate(H0, XBar{rho_indx}, S, N0);
    
        rd = pskdemod(H0*XBar{rho_indx} + W, Q, pi/Q);
        [~, errRate] = symerr(data, rd);
        RealSymErrRateNominal(rho_indx, mc)   = errRate;
        
        %{ 
            % NB: No value to study the optimal waveforms because they are specific to true unknown channels
            &
            % Optimal Waveform
            % "PowerType" have already been validated before; so no need for further verification
            switch PowerType
                case 'TPC'
                    [~, X0] = g1(H0, S, Xs, Pt, rho);
                case 'PAPC'
                    [~, X0] = g2(H0, S, Xs, Pt, rho);
            end
            GetAvrgSumRate(H0, X0, S, N0)
        
            rd = pskdemod(H0*X0 + W, Q, pi/Q);
            [~, errRate] = symerr(data, rd);
            errRate
        %}
    end

    RealSumRateRobust{rho_indx}     = RealSumRateRobust_Local;
    RealSymErrRateRobust{rho_indx}  = RealSymErrRateRobust_Local;
end
%---------------------  End of WaveformDesign_Joint -------------------------


