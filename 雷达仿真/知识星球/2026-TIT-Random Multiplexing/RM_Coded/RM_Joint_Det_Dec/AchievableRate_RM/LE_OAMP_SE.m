 function  v_post= LE_OAMP_SE(v, dia, v_nn, M, N)
 if N>M
    dia=dia.';
 end
v_post=zeros(length(v_nn),size(v,2));
for jj=1:length(v_nn)
    v_n=v_nn(jj);
    for i=1:length(v)
    rho = v_n / v(i);
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    v_post(jj,i) = v_n / N * sum(D);
    end  
end
 end