function C=k_r_prod(A,B)
[M1,N1]=size(A);
[M2,N2]=size(B);
if N1==N2
    C=zeros(M1*M2,N1);
    for i=1:N1
        C(:,i)=kron(A(:,i),B(:,i));
    end
end
end