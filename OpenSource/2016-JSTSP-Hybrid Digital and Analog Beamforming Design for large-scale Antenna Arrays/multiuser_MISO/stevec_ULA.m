function a = stevec_ULA(theta, M)
    % Generates a steering vector for Uniform Linear Array (ULA)
    % theta: rad
    m = 0:M-1;
    a = exp(1i * pi * m * sin(theta))/sqrt(M);
    a = a.';
end
