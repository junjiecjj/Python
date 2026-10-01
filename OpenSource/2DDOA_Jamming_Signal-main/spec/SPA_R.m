function R=SPA_R(Sa_set,Ri,Mf1,Mf2)
    %%基本参数
    J=J_create(Mf1,Mf2);
    Sa_sk=logical(kron(Sa_set,Sa_set));
    Js=J(Sa_sk,:);
    %%二次cost参数[u;sigma]'*[A,p;p',d]*[u;sigma]-2real([b1;b2]'[u;sigma])
    A0=Js'*kron(Ri.',Ri)*Js;
    b=Js'*vec(Ri);
    u=pinv(A0)*b;
    R=B_toepw(u,Mf1,Mf2);
end