%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function H = ComplexMatrixize(h, K, N)
% Matrixize a real vector into a K-by-N complex matrix
    len = length(h); 
    assert(len == K*N*2);

    real_h = h(1:K*N);          % real part
    imag_h = h(K*N+1 : end);    % imaginary part

    H = reshape(real_h, K, N) + 1j * reshape(imag_h, K, N);
end