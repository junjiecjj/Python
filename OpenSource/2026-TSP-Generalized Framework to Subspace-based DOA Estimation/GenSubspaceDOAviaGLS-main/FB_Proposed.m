function [FB_PM_DOA_degree,FB_PM_DOA_radian] = FB_Proposed(R,K,M,gg,x0,QQ,Q,N)


PI_N=zeros(N,N);
for ii=1:N
    PI_N(ii,N+1-ii)=1;
end

PI_M=zeros(M,M);
for ii=1:M
    PI_M(ii,M+1-ii)=1;
end


X_bar=[((Q+PI_M*Q*PI_M)^(-0.5))*x0   PI_M*((Q+PI_M*Q*PI_M)^(-0.5))*conj(x0)*PI_N];
[US,D0,~]=svd(X_bar,'econ');
US=US(:,1:K);
dd=D0(1:K,1:K);


US=((Q+PI_M*Q*PI_M)^(0.5))*US;
scnum=K;
ind=[];
U_bar = zeros(M,K);
for ii=1:K
U_bar(:,ii) = fft(US(:,ii));
end

WL=zeros(M,scnum);
WLL=zeros(M,scnum);
kk=0:M-1;
idx=1:scnum;
for ii=1:M
WL(ii,:)=exp(-1j*2*pi*kk(ii)*idx/M);
WLL(ii,:)=exp(-1j*2*pi*kk(ii)*(idx-1)/M);
end
WW=zeros(M,M);
idx=0:M-1;
for ii=1:M
WW(ii,:)=exp(-1j*2*pi*kk(ii)*idx/M);
end

for zz=1:length(QQ)
H=[];
h=[];
[~,s1]=maxk(abs(U_bar(:,1)),QQ(zz));
Z1=zeros(QQ(zz),M);
for ii=1:QQ(zz)
Z1(ii,s1(ii))=1;
end
B=null((Z1*WLL).');
B_her=B.';
for ii=1:K
S1=U_bar(s1,ii);
T1=diag(S1);
H=[H;B_her*T1*WL(s1,:)];
h=[h;-B_her*S1];
end
a=H\h;
CC=B_her*(eye(QQ(zz))+diag(WL(s1,:)*a))*WW(s1,:);
CC=inv(CC*(Q+PI_M*Q*(PI_M'))*(CC'));
WA=kron(dd^2,CC);
for ii=1:gg
a=(H'*WA*H)\(H'*WA*h);
CC=B_her*(eye(QQ(zz))+diag(WL(s1,:)*a))*WW(s1,:);
CC=inv(CC*(Q+PI_M*Q*(PI_M'))*(CC'));
WA=kron(dd^2,CC);
end
v=roots([1;a]);
C1=angle(v);
C2=asin(-C1/pi);

ind=[ind;C2];



end

ind22=ind(K+1:end);



theta=-(pi/2):0.01:(pi/2);
[ind1] = cb_pre(R, M, theta,ind,K);



if length(ind1) >= K
ind=ind1;
else
ind=ind22;  
end




[ind3,ind4] = GLR(R, ind,M,Q);
ind=ind4;


idx_combinations = nchoosek(1:length(ind), K-1 );

n = size(idx_combinations,1); % number of combinations.


R_tilde = (Q^(-0.5))*R*(Q^(-0.5));
idxR = (0 : (M - 1))';
Ahat = zeros(M, K-1);
ML_objective = zeros(n, 1);

for i = 1:n
    idx_tmp =  idx_combinations(i,:);
    
    for k = 1:K-1
        Ahat(:,k) = exp(-1j*pi*sin(ind(idx_tmp(k)))*idxR);
    end
    Ahat = (Q^(-0.5))*Ahat;
    Ahat = Ahat/sqrt(M); % So that each column of Ahat has unit norm.
    PAhat = Ahat*((Ahat')*Ahat)^(-1)*(Ahat');
    PA_orthhat = eye(M,M) - PAhat;
    aaa1=exp(-1j*pi*sin(ind3)*idxR);
    nu1= (PA_orthhat*(Q^(-0.5))*aaa1)/norm(PA_orthhat*(Q^(-0.5))*aaa1);
    ML_objective(i) =  real( trace(  ( (PA_orthhat-nu1*(nu1')) * R_tilde ) ));

       
end

[~,idx] = min(ML_objective);
idx_tmp =  idx_combinations(idx,:);
FB_PM_DOA_radian = sort([ind(idx_tmp); ind3]);
FB_PM_DOA_degree = FB_PM_DOA_radian*(180/pi);

end




%% Functions Used
function [ind1] = cb_pre(R, M, theta,ind,K)
ind1=[];
idxm = (0:(M-1))'; 
GridSize = length(theta);
f_CBF = zeros(GridSize,1);
for i = 1:GridSize
    a = (1/sqrt(M))*exp(-1j*pi*sin(theta(i))*idxm);
    f_CBF(i) = real((a')*R*a);
end
[PKS,~]= findpeaks(f_CBF,'SortStr','descend','NPeaks',K+1);
ff=PKS(end);
for ii=1:length(ind)
    a = 1/sqrt(M)*exp(-1j*pi*sin(ind(ii))*idxm);
    b= real((a')*R*a);
    if b >= ff
        ind1=[ind1;ind(ii)];
    end
end

if length(ind1) == (2*K)
ind1=ind(end-(K-1):end);
end
if length(PKS)< K+1
ind1=ind(end-(K-1):end);
end


end




function [ind2,ind3] = GLR(R, ind,M,Q)
ind2=[];
idxR = (0 : (M - 1))';
R=(pinv(Q)^(0.5))*R*(pinv(Q)^(0.5));
P_orth=eye(M);
EW=zeros(size(ind));
for k=1:length(ind)
        asteer = (pinv(Q)^(0.5))*exp(-1j*pi*sin(ind(k))*idxR);
        EW(k) = ((asteer')*P_orth*R*P_orth*asteer) / ((asteer')*P_orth*asteer);
end
    
[~,I] = max(EW);
ind2=[ind2;ind(I)];
ind(I) = [];
ind3=ind;

end