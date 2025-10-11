function gdop = doa_tdoa_gdop(S0, S1, X, sigma_angle, sigma_S, sigma_t)
    x = X(1); y = X(2); x0 = S0(1); y0 = S0(2); x1 = S1(1); y1 = S1(2);
    c = 3*10^8;
    C = zeros(2); U = zeros(2); W = zeros(2);
    sigma_r = c * sigma_t;
    r1 = sqrt((x - x1)^2 + (y - y1)^2);
    r0 = sqrt((x - x0)^2 + (y - y0)^2);
    sin0 = (y - y0)/r0; cos0 = (x - x0)/r0;
    C = [-sin0^2/(y - y0), cos0^2/(x - x0); (x - x1)/r1 - (x - x0)/r0, (y - y1)/r1 - (y - y0)/r0];
    U = [sin0^2/(y - y0), -cos0^2/(x - x0); (x - x0)/r0, (y - y0)/r0];
    W = [0, 0; (x - x1)/r1, (y - y1)/r1];
    Rv = [sigma_angle^2, 0; 0, sigma_r^2];
    Rs = [sigma_S^2, 0; 0, sigma_S^2];
    Pdx = pinv(C) * (Rv + U * Rs * U' + W * Rs * W') * pinv(C)';
    gdop = sqrt(Pdx(1,1) + Pdx(2,2));
end