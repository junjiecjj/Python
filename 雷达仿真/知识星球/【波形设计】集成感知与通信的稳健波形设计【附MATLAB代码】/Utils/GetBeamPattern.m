%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function B = GetBeamPattern(R, A_SteerVector)
    % Compute the resulting beam pattern given the covariance matrix
    B = abs(diag(A_SteerVector'*R*A_SteerVector))/(4*pi);
end