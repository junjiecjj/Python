%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function [val, X] = g1(H, S, Xs, Pt, rho)
    % Evaluate the cost for the waveform X at the channel H
    [N, L] = size(Xs);
    
    % A = [sqrt(rho)*H; sqrt(1-rho)*eye(N,N)];
    % B = [sqrt(rho)*S; sqrt(1-rho)*X0];
    % Q = A'*A;
    % G = A'*B;
    
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