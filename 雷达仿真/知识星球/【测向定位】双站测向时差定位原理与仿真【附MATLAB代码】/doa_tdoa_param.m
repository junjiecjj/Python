function [delta_t, tan0, tan1] = doa_tdoa_param(S0, S1, X)
    c = 3*10^8;
    r0 = sqrt((X(1) - S0(1))^2 + (X(2) - S0(2))^2);
    r1 = sqrt((X(1) - S1(1))^2 + (X(2) - S1(2))^2);
    delta_t = (r1 - r0)/c;
    tan0 = (X(2) - S0(2))/(X(1) - S0(1));
    tan1 = (X(2) - S1(2))/(X(1) - S1(1));
end