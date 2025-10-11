%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function Rmmse = GetR(normalizedPos, Bdes, ang, Pt, N)
    % Solve the optimization problem to find the covariance matrix
    Rmmse = helperMMSECovariance(normalizedPos, Bdes, ang);
    
    % helperMMSECovariance returns a covariance matrix with ones along the
    % diagonal such that the total transmit power is equal to N. Renormalize
    % the covariance matrix to make the total transmit power equal to Pt.
    Rmmse = Rmmse*Pt/N;

    % If a bad situation happens
    % assert(min(svd(Rmmse)) >= 1e-6);
end