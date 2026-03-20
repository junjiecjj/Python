function [v_part2]=Opt_MMSE(x, v_DE, rho_LE, v_LE, phi_1)

x_tmp=x(x>phi_1); 
v_LE_inv=zeros(1,length(x_tmp));
%% fit inverse function of GLE
for i=1:length(x_tmp)
    id_max=min(find(rho_LE<x_tmp(i)));
    if id_max==1
        v_LE_inv(i)=1e-5;
    else
        id_min=max(find(rho_LE>x_tmp(i)));
        v_LE_inv(i)=v_LE(id_min)+((v_LE(id_max)-v_LE(id_min))/(rho_LE(id_max)-rho_LE(id_min)))*(x_tmp(i)-rho_LE(id_min));
    end
end
v_LE_inv(v_LE_inv>1)=1.0;
v_part2=min(v_LE_inv, v_DE);

end