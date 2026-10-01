function [D_omega,D_gamma]=D2MUSIC1(Rx,fun_A,K,omega0,gamma0,isplot)
U=size(Rx,1);
[V,D]=eig(Rx);
[~,I]=sort(diag(D));
V=V(:,I);
Un=V(:,1:U-K);

n1=500;
n2=500;
omega=pi*linspace(-1,1,n1+1);
gamma=pi*linspace(-1,1,n2+1);
omega=omega(1:n1);
gamma=gamma(1:n2);
G=zeros(n1,n2);
for i=1:n1
    for j=1:n2
        a=fun_A(omega(i),gamma(j));
        G(i,j)=1./norm(a'*Un,2);
    end
end
LL=imregionalmax(G);
LL([1,end],:)=0;
LL(:,[1,end])=0;
[row,col] = find(LL);
[X,Y]=meshgrid(gamma,omega);
if isplot
    h=figure;
    G0=db(G/max(max(abs(G))));
    G0=G0/abs(min(min(G0)))*30;
    
    mesh(Y,X,G0)
    hold on;
    hh=plot3(omega0,gamma0,zeros(1,K),'m*','Markersize',8);
    xlabel('\omega');
    ylabel('\gamma');
    zlabel('Spectrum (dB)')
    %     colormap(parula(5))
    legend(hh,'True')
    xlim([-pi,pi]);
    ylim([-pi,pi]);
    
    %     colormap(jet)
    set(gca,'fontsize',16,'fontname','Times');
    set(h,'position',[100 200 600 450]);
    
end
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