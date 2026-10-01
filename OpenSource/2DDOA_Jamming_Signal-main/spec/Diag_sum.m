function u=Diag_sum(A)
    [M,~]=size(A);
    u=zeros(2*M-1,1);
    for i=1-M:M-1
        u(M+i)=sum(diag(A,i));
    end
end