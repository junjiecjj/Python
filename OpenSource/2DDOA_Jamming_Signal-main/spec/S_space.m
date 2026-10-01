function Us=S_space(Rx,d)
    [V,D]=eig(Rx);
    [~,I]=sort(diag(D),'descend');
    V=V(:,I);
    Us=V(:,1:d);
end