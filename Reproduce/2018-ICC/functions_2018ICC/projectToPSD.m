function R_psd = projectToPSD(R)
    R = (R + R') / 2;
    [V, D] = eig(R);
    lambda = real(diag(D));
    lambda(lambda < 0) = 0;
    R_psd = V * diag(lambda) * V';
    R_psd = (R_psd + R_psd') / 2;
end
