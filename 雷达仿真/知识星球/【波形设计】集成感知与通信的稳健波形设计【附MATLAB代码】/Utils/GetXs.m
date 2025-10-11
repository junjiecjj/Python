%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function Xca = GetXs(Rmmse, L)
    eta = 1.1;                          % Parameter that controls low PAR constraint
    
    % Find a set of waveform with the covariance equal to Rmmse using the cyclic algorithm (CA). The length of each waveform is M.
    Xca = helperCAWaveformSynthesis(Rmmse, L, eta);
end

