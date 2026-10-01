function [c,d]=CRB_suc(dx,A,P,W,N)
[M,K]=size(A);
R=A*P*A'+W*eye(M);
D=1j*[dx.*A(1:M/2,:),zeros(M/2,K);dx.*A(M/2+1:end,:),A(M/2+1:end,:)];
Ak=k_r_prod(conj(A),A);
Ad=k_r_prod(conj(D),[A,A])+k_r_prod(conj([A,A]),D);
[U,S]=eig(R);
G=diag(1./sqrt(diag(S)));
Rh=U*G*U';
Rhk=kron(Rh.',Rh);
B=Rhk*Ad*[P,zeros(K);zeros(K),P];
Tau=Rhk*[Ak,vec(eye(M))];
c0=1/N*(real(B'*(eye(M^2)-Tau*pinv(Tau))*B))^-1;
c=(trace(c0(1:K,1:K)))/K;
d=((trace(c0(K+1:end,K+1:end))))/K;
end

