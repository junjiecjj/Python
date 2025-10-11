function alpha = OPT_REFL_COEFF (H_dash, R, T_dash, alpha, No, Nr, M);
for m=1:M
    C=0;
    for m1 = 1:M
        if m1 ~= m
            C=C+alpha(m1)*R(:,m1)*(T_dash(m1,:));
        end
    end
    A = eye(Nr) + (H_dash + C)*(H_dash + C)'/No + R(:,m)*(T_dash(m,:))*(R(:,m)*(T_dash(m,:)))'/No;
    B = R(:,m)*(T_dash(m,:))*(H_dash + C)'/No;
    if trace((A\B))~=0
        alpha(m) = exp(-1j*angle(max(eig(A\B))));
    else
        alpha(m) = 1;
    end
end