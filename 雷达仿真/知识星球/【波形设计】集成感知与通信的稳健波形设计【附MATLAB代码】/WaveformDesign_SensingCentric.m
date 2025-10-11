%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author: Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:   1 September 2023, 21 March 2024
    @Home:   https://github.com/Spratm-Asleaf/Robust-Waveform

    @Acknowledgement: https://mathworks.com/help/phased/ug/waveform-design-for-a-dual-function-mimo-radcom-system.html
    @Dependencies: Phased Array System Toolbox, Communications Toolbox, Signal Processing Toolbox, DSP System Toolbox
%}

% Nominal Design
tic;
[~, XBar]          = f(HBar, S, F);
disp('            ----------');
disp(['            Time: ' num2str(toc) '(s)']);
disp('            ----------');

% Robust Design; every "theta" gives different result
NumTheta = 50;                                  % How many "theta" to be tested for each episode
Theta = linspace(0, epsilon*4, NumTheta);     % Compared to "epsilon", "theta" should not be overly large
Xr = cell(NumTheta, 1);                         % Robust Waveform against "theta"
Hr = cell(NumTheta, 1);                         % Worst-case channel associated with Robust Waveform Xr
TotalRunningTime = 0;
disp('      Robust Design:');
for i = 1:NumTheta
    tic;
    switch DesignMethod
        case 1
            [~, Xr{i}, Hr{i}] = GetRobustSolution_Method1(HBar, S, F, Theta(i));          % Method 1    % running time: 5.83 ms
        case 2
            [~, Xr{i}, Hr{i}] = GetRobustSolution_Method2(HBar, XBar, S, F, Theta(i));    % Method 2    % running time: 4.60 ms
        otherwise
            error('WaveformDesign_SensingCentric :: Error in Sensing-Centric Design Method :: Non-exist!');
    end

    TotalRunningTime = TotalRunningTime + toc;
    disp(['            Trying theta = ' num2str(Theta(i)) ' *** Current Progress: ' num2str(100*i/NumTheta) '%']);
end
disp('            ----------');
disp(['            Time (Average): ' num2str(TotalRunningTime/NumTheta) '(s)']);
disp('            ----------');

%% Assign Storage
MC = 1000;                                          % How many Monte-Carlo episodes (aka: trials, scenarios) to run

% Achievalbe Sum Rate
NominalSumRate        = zeros(1, 1);                % The estimated sum_rate claimed by nominal solutions
RealSumRateNominal    = zeros(MC, 1);               % The real sum_rate evaluated with nominal solutions

RobustSumRate         = zeros(1, NumTheta);         % The estimated sum_rate claimed by robust solutions
RealSumRateRobust     = zeros(MC, NumTheta);        % The real sum_rate evaluated with robust solutions

% Symbol Error Rate
RealSymErrRateNominal = zeros(MC, 1);
RealSymErrRateRobust  = zeros(MC, NumTheta);

%% Main Simulation Loop
NominalSumRate        = GetAvrgSumRate(HBar, XBar, S, N0);
for i = 1:NumTheta
    RobustSumRate(i)  = GetAvrgSumRate(Hr{i}, Xr{i}, S, N0);
end
disp('Monte-Carlo Waveform Testing ... (Stay Patient! This is quick!)');
for mc = 1:MC
    % Real channel varies from one realization to another
    H0 = HRef + ((2*rand(size(HRef)) - 1) + (2*rand(size(HRef)) - 1) * 1j)*epsilon;
    W = sqrt(N0)*GetChannelNoise(K, L, 'GAUSSIAN');         % Channel noise
    % NB: The actual received signal is Y = H0*X + W, given waveform X
    
    %% Waveform Evaluation
    % Robust Waveform
    for i = 1:NumTheta
        RealSumRateRobust(mc, i)      = GetAvrgSumRate(H0, Xr{i}, S, N0);

        rd = pskdemod(H0*Xr{i} + W, Q, pi/Q);                                   
        [~, errRate] = symerr(data, rd);
        RealSymErrRateRobust(mc, i)   = errRate;
    end
    
    % Nominal Waveform
    RealSumRateNominal(mc)     = GetAvrgSumRate(H0, XBar, S, N0);

    rd = pskdemod(H0*XBar + W, Q, pi/Q);
    [~, errRate] = symerr(data, rd);
    RealSymErrRateNominal(mc)   = errRate;
    
    %{
        % NB: No value to study the optimal waveforms because they are specific to true unknown channels
        %
        % Optimal Waveform
        [~, X0]          = f(H0, S, F);
        GetAvrgSumRate(H0, X0, S, N0);
    
        rd = pskdemod(H0*X0 + W, Q, pi/Q);
        [~, errRate] = symerr(data, rd);
        errRate;
    %}
end
%---------------------  End of WaveformDesign_SensingCentric -------------------------


