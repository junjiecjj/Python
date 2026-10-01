function [D_omega,D_gamma]=D2MUSIC(Rx,funA,K)
U=size(Rx,1);
[V,D]=eig(Rx);
[~,I]=sort(diag(D));
V=V(:,I);
Un=V(:,1:U-K);

n1=400;
n2=200;
omega=pi*linspace(-1,1,n1+1);
gamma=pi*linspace(-1,1,n2+1);
omega=omega(1:n1);
gamma=gamma(1:n2);
G=zeros(n1,n2);
for i=1:n1
    for j=1:n2
        a=funA(omega(i),gamma(j));
        G(i,j)=1./norm(a'*Un,2);
    end
end
LL=imregionalmax(G);
LL([1,end],:)=0;
LL(:,[1,end])=0;
[row,col] = find(LL);
%     [X,Y]=meshgrid(gamma,omega);
%     figure;
%     plot3(Y,X,G)
%     xlabel('omega');
ind=sub2ind([n1,n2],row,col);
v=G(ind);
[~,ii]=sort(v,'descend');
row=row(ii);
col=col(ii);
v=v(ii);
if length(row)<K
    oo=ceil(K/length(row));
    row=repmat(row,oo,1);
    col=repmat(col,oo,1);
    v=repmat(v,oo,1);
end
row=row(1:K);
col=col(1:K);
v=v(1:K);
[D_omega,kk]=sort(omega(row));
D_gamma=sort(gamma(col(kk)));
end