function D = Dmatrix(K)
    var_nr = (10.^(8/10))^2; mean_nr = 3;
    mu = log10(mean_nr^2/sqrt(var_nr+mean_nr^2)); sigma_nr = sqrt(log10(var_nr/(mean_nr^2+1)));
    nr = lognrnd(mu,sigma_nr,[1,K]);
    dr = randi([100,1000],1,K)/100;
    D = diag(nr./dr.^3.8);