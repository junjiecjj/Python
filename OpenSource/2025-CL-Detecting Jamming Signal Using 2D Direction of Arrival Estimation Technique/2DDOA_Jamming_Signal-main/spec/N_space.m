function Un=N_space(Rx,d)
    [V,D]=eig(Rx);
    [~,I]=sort(diag(D),'descend');
    V=V(:,I);
    Un=V(:,d+1:end);
end