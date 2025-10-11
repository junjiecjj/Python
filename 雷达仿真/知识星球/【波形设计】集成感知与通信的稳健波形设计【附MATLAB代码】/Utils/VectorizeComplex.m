%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function h = VectorizeComplex(H)
% Vectorize a complex matrix into a column real vector
    real_H = real(H);
    imag_H = imag(H);
    h = [real_H(:); imag_H(:)];
end