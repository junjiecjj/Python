function vmmse=GOAMP_GLE_inv_SE(v_LE, vpre_LE, vpost_LE)

vmmse=zeros(1, length(v_LE));
%% fit inverse function of GLE
for i=1:length(v_LE)

    id_max=max(find(vpost_LE<v_LE(i)));

    id_min=min(find(vpost_LE>v_LE(i)));

    if (isempty(id_min))
        vmmse(i)=vpre_LE(end);
    elseif (isempty(id_max))
         vmmse(i)=vpre_LE(id_min);
    else
        vmmse(i)=vpre_LE(id_min)+((vpre_LE(id_max)-vpre_LE(id_min))/(vpost_LE(id_max)-vpost_LE(id_min)))*(v_LE(i)-vpost_LE(id_min));
    end
end
end