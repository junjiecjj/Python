%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function [val, X] = g2(H, S, Xs, Pt, rho)
    % Evaluate the cost for the waveform X at the channel H

    X = helperRadComWaveform(H, S, Xs, Pt, rho);        % I suspect whether this guy is reliable? I am not sure.

    val = rho * norm_square(H*X - S, 'fro') + (1 - rho)*norm_square(X - Xs, 'fro');
end