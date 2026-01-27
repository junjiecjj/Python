%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function [val, X] = f(H, S, F)
    % Evaluate the cost for the waveform X at the channel H
    [U, ~, V] = svd(F'*H'*S);
    [N, ~] = size(U);
    [L, ~] = size(V);
    
    if N >= L
        X = sqrt(L)*F*U*[eye(L, L); zeros(N-L, L)]*V';
    else
        X = sqrt(L)*F*U*[eye(N, N), zeros(N, L-N)]*V';
    end
    
    val = norm_square(H*X - S, 'fro');
end