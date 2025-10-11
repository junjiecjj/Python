%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function [robust_cost, Xr, HStar] = GetRobustSolution_Method3(HBar, XBar, S, theta, Xs, Pt, rho, PowerType)
    %% Implementation of Method 3

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
    alpha = 100;            % Use 10000 also ok, no much different effects
    H_ = [
        sqrt(rho) * HStar
        sqrt(1-rho) * eye(N)
    ];

    S_ = [
        sqrt(rho) * S
        sqrt(1-rho) * Xs
    ];

    %% Joint Design
    PowerType = upper(PowerType);
    switch PowerType
        case 'TPC'  % Total Power Constraint (TPC)
            [robust_cost, Xr] = g1(H_, S_, XBar, Pt, alpha/(alpha + 1));
        case 'PAPC'  % Per-Antenna Power Constraint (PAPC)
            alpha = 10;   % The method is computationally heavy, especially when conducting robust design.
                          % For short running times, the total iteration step is limited, which compromises the optimality a lot.
            [robust_cost, Xr] = g2(H_, S_, XBar, Pt, alpha/(alpha + 1));
        otherwise
            error('GetRobustSolution_Method3 :: Error in Power-Constraint Type :: Non-exist!');
    end
end