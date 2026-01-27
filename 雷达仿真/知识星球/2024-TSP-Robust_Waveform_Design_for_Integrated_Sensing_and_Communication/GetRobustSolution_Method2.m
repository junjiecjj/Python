%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function [robust_cost, Xr, HStar] = GetRobustSolution_Method2(HBar, XBar, S, F, theta)
    %% Implementation of Method 2

    %% Initialization
    [K, N] = size(HBar);

    C__ = kron(XBar.', eye(K));
    Gamma = real(C__); 
    Theta = imag(C__);
    C = [Gamma -Theta; Theta Gamma];
    s    = VectorizeComplex(S);
    hBar = VectorizeComplex(HBar);

    %% Get Worst-Case Channel
    % Proposation 5
    hStar = ConvexQuadraticMaximization(C, s, hBar, theta);     
    HStar = ComplexMatrixize(hStar, K, N);

    %% Get Robust Waveform
    % Proposition 9
    alpha = 10000;
    H_ = [
        sqrt(alpha) * HStar
        eye(N)
    ];

    S_ = [
        sqrt(alpha) * S
        XBar
    ];

    [robust_cost, Xr] = f(H_, S_, F);
end