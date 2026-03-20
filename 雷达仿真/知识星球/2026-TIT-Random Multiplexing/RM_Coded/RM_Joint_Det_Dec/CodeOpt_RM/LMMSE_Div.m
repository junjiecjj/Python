function [rho] = LMMSE_Div(v_in, dia, K, Sigma_n)

L_s = length(Sigma_n);
L_v = length(v_in);
rho = zeros(L_s, L_v);

for i=1:L_s
    for j=1:L_v 
        v_lmmse = 1./( Sigma_n(i)^-1* v_in(j) * dia.^2 + 1); 
        rho(i,j) = v_in(j)^-1 * (K/sum(v_lmmse) - 1);  
    end
end

end

