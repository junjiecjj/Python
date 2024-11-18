function [hatx_array, rz_array, hatv_array]=Mdamping(hatx_array, rz_array, hatv_array, t)
L=2;
lt=min(L, t);
Temv=hatv_array(t-lt+1:t, t-lt+1:t);

[~,b]=chol(Temv);

if b==0
    zeta=Temv^(-1)*ones(lt,1);
    v_phi=1/sum(zeta);
else
    v_phi=hatv_array(t,t);
    zeta=[zeros(lt-1,1);1];
end
zeta=zeta*v_phi;

hatv_array(t,t)=v_phi;

hatx_array(:,t)=hatx_array(:,t-lt+1:t)*zeta; 
rz_array(:,t)=rz_array(:,t-lt+1:t)*zeta;

for ii=1:t-1
   hatv=0;
   for jj=1:lt
       hatv=hatv+zeta(jj)*hatv_array(t-lt+jj,ii);
   end
   hatv_array(t,ii)=hatv;
   hatv_array(ii,t)=hatv;
end

end