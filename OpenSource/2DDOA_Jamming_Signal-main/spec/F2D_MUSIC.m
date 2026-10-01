function [M_omega,M_gamma]=F2D_MUSIC(Rx,U,K)
[V,D]=eig(Rx);
[~,I]=sort(diag(D));
V=V(:,I);
Un=V(:,1:2*U-K);

Un1=Un(1:U,:);
Un2=Un(U+1:end,:);
syms z;
pz=z.^(0:U-1).';
pz1=z.^(U-1:-1:0);
fz11=pz1*(Un1*Un1')*pz;
fz12=pz1*(Un1*Un2')*pz;
fz21=pz1*(Un2*Un1')*pz;
fz22=pz1*(Un2*Un2')*pz;
fz=fz11*fz22-fz12*fz21;
aa=sym2poly(fz);
zx=roots(aa);
zx=zx(abs(zx)<=1);
% 1-abs(zx)
% angle(zx)
[~,zo]=sort(1-abs(zx));
zx=zx(zo(1:K));
M_omega=sort(angle(zx)).';
M_gamma=zeros(1,K);
for k=1:K
    at=A_th((0:U-1)',M_omega(k));
    M_gamma(k)=angle(-at'*Un2*Un1'*at);
end
end