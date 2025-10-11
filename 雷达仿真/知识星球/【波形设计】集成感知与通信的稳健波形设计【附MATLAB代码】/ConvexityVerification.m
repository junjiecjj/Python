%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     5 May 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform

    @Acknowledgement: https://mathworks.com/help/phased/ug/waveform-design-for-a-dual-function-mimo-radcom-system.html
    @Dependencies:    Phased Array System Toolbox, Communications Toolbox, Signal Processing Toolbox, DSP System Toolbox
%}


%% Dependencies
% Externel
addpath('.\Externals');
% Internel
addpath('.\Utils');

%% To test the convexity and concavity of the function f(H)
%  NB: Construct couterexamples to show f(H) is neither convex nor concave
clc;

I = 1000;           % Among "I" realizations
epsilon = 0.05;
isConvex = zeros(I, 1);
Hbar = 0.9 + 1j*0.5;
S = 1 + 1j;
F = 1;              % F*F' = R

for i = 1:I
    H1 = Hbar + sqrt(0.5)*(randn + randn*1j)*epsilon;
    H2 = Hbar + sqrt(0.5)*(randn + randn*1j)*epsilon;
    alpha = 0.5;

    if     f(alpha*H1 + (1-alpha)*H2, S, F) <= alpha*f(H1, S, F) + (1-alpha)*f(H2, S, F)    % Convex
        isConvex(i) = 1;
    elseif f(alpha*H1 + (1-alpha)*H2, S, F) >= alpha*f(H1, S, F) + (1-alpha)*f(H2, S, F)    % Concave
        isConvex(i) = 0;
    end
end

ConvexFlag  = find(isConvex == 1);
ConcaveFlag = find(isConvex == 0);



