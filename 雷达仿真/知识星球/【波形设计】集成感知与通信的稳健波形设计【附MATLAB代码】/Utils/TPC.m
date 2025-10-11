%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function X = TPC(Xc, Xs, Pt, rho)
% Solution to Total-Power Constrained (TPC) Waveform Synthesis
%       min   rho * || X - Xc ||^2_F + (1 - rho) * || X - Xs ||^2_F
%       s.t.  trace(X*X'/L) = Pt
% where Xc is the perfect waveform for communication and Xs is the perfect waveform for sensing

    [N, L] = size(Xs);
    [a, b] = size(Xc);
    assert(N == a && L == b);
    
    X = JointWaveformDesign_TPC(eye(N), Xc, Xs, Pt, rho);
end

function [X, val] = JointWaveformDesign_TPC(H, S, Xs, Pt, rho)
% This function solves (16) of [Liu et al; 2018; TSP; 10.1109/TSP.2018.2847648]

    % Evaluate the cost for the waveform X at the channel H
    [N, L] = size(Xs);
    
    % A = [sqrt(rho)*H; sqrt(1-rho)*eye(N)];
    % B = [sqrt(rho)*S; sqrt(1-rho)*Xs];
    % Q = A'*A;
    % G = A'*B;
    % Can you see the benefit of the following alternative definitions of Q and G? Hint: what if rho = 0 or rho = 1?
    
    Q = rho*(H'*H) + (1-rho)*eye(N);
    G = rho*H'*S + (1-rho)*Xs;

    lambda_min = min(eig(Q));
    func = @(lambda) norm_square((Q + lambda*eye(N, N))^-1*G, 'fro') - L*Pt;
    lambda = fzero(func, -lambda_min);
    assert(lambda >= -lambda_min);
    assert(func(lambda) <= 1e-6);
    X = (Q + lambda*eye(N, N))^-1*G;

    val = rho * norm_square(H*X - S, 'fro') + (1 - rho)*norm_square(X - Xs, 'fro');
end

function val = norm_square(X, type)
    val = (norm(X, type))^2;
end