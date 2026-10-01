function c=N_est(A,K)
    A=(A+A')/2;
    N=size(A,1);
    [~,D]=eig(A);
    d=diag(D);
    [d,~]=sort(d,'descend');
    c=sum(d(K+1:end))/(N-K);
end