function B=mat_sqrt(A)
    [U,D]=eig(A);
    B=U*sqrt(abs(D))*U';
end