function v_out = LMMSE1(rho, M,N, dia, Sigma_n)
rho=0.4750;
vpre_LE=[1e-6:1e-4:0.1, 0.1:0.01:1];
%vpost_LE=GOAMP_GLE_Clip_SE(delta, clip, snr.^-1, vpre_LE);
vpost_LE= LE_OAMP_SE(vpre_LE, dia, Sigma_n, M, N);
%plot(vpre_LE,vpost_LE )
v_LE    =[1e-6:1e-4:0.1, 0.1:0.01:1];

v_mmse = GOAMP_GLE_inv_SE(v_LE, vpre_LE, vpost_LE(:));
%plot(v_LE,v_mmse )
rho_LE = 1./v_LE- 1./v_mmse;

plot(rho_LE,v_LE )


phi_1 =rho_LE(:,end);

[v_out]=Opt_MMSE(rho, rho_LE, v_LE, phi_1);
end


function [v_LE_inv]=Opt_MMSE(x, rho_LE, v_LE, phi_1)

x_tmp=x(x>phi_1); 
v_LE_inv=zeros(1,length(x_tmp));
%% fit inverse function of GLE
for i=1:length(x_tmp)
    id_max=min(find(rho_LE<x_tmp(i)));
    if id_max==1
        v_LE_inv(i)=1e-5;
    else
        id_min=max(find(rho_LE>=x_tmp(i)));
        v_LE_inv(i)=v_LE(id_min)+((v_LE(id_max)-v_LE(id_min))/(rho_LE(id_max)-rho_LE(id_min)))*(x_tmp(i)-rho_LE(id_min));
    end
end
v_LE_inv(v_LE_inv>1)=1.0;
v_part2=min(v_LE_inv);

end


function  v_post= LE_OAMP_SE(v, dia, v_n, M, N)
v_post=zeros(size(v,1),size(v,2));
    for i=1:length(v)
    rho = v_n / v(i);
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    v_post(i) = v_n / N * sum(D);
    end
end