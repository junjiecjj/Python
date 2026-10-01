function [omega,gamma]=Noise_pencil(R,k)
    [V,D]=eig(R);
    d=diag(D);
    oi=find(d>0);
    if k<length(oi)
        [~,oj]=sort(d(oi));
        oi=oi(oj(end-k+1:end));
    end
    dp=sqrt(d(oi));
    Vp=V(:,oi);
    L0=Vp*diag(dp);
    X=L0(1:end/2,:);
    Y=L0(end/2+1:end,:);
    [V1,~,V2]=svd(X'*Y);
    U=V1*V2';
    L=1/2*(X+Y*U');
    [G,S]=eig(U);
    gamma=angle(diag(S));
    L=L*G;
    omega=angle(diag(pinv(L(1:end-1,:))*L(2:end,:)));
    [omega,ii]=sort(omega.');
    gamma=gamma(ii).';
    
end