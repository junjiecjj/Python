function  [IP_oUega,IP_gaUUa]=IP_PU(Rx,U,K)
Un=N_space(Rx,K);
ne=trace((Un'*Rx*Un))/(2*U-K);

Ru11 = Rx(1:U,1:U);
Ru22 = Rx(U+1:end,U+1:end);
Ru21 = Rx(U+1:end,1:U);

[U,~]=size(Ru22);
R1=([Ru21;Ru22-ne*eye(U)]);
R2=([Ru11-ne*eye(U);Ru21']);

PU1=R1*pinv(R2);
[V,D]=eig(PU1);
d=diag(D);
[~,I]=sort(abs(d),'descend');
d=d(I(1:K));
gaUUa0=angle(d).';

V=V./(V(1,:));
V1=V([1:U-1,U+1:end-1],I(1:K));
V2=V([2:U,U+2:end],I(1:K));
PU2=pinv(V1)*V2;
oUega0=angle(diag(PU2));
[IP_oUega,ii]=sort(oUega0.');
IP_gaUUa=gaUUa0(ii);
end