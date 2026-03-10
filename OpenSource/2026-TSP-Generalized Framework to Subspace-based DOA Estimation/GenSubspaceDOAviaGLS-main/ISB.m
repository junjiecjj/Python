function [Q] = ISB(R,K,M,eps)

Q=diag(diag(R));
count = 0;
zez = 0;

while zez == 0
count = count+1;
[UU,DDDD] = eig(R,Q);
[~,idx] = sort(real(diag(DDDD))); 
UU1 = UU(:,idx);
UU = UU1(:,1:(M-K));
UU = UU/norm(UU,'fro');

VU = UU;
UU1 = UU*(UU');
UU = R*UU1 + UU1*R;
fgf = rem(count,M)+1;
f0= norm((R-Q)*VU,'fro')^2;

Q_1=0.5*diag(diag(UU))*diag(1./diag(UU1));
Q(fgf,fgf)=Q_1(fgf,fgf);

f1 = norm((R-Q)*VU,'fro')^2;
if  abs(f0-f1) < eps
    zez =1;
end
end


end

