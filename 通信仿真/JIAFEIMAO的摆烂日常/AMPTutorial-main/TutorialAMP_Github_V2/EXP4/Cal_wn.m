function [w, w0, over_w, over_w00,  over_wt0]=Cal_wn6(Input, lambda, lambda_dag, IterNum)
M=Input.M;
N=Input.N;
w0=1/N*sum(lambda);
w=zeros(2*IterNum+1,1);
Tem=ones(M,1);
for ii=1:2*IterNum+1
    Tem=Tem.*(lambda_dag*ones(M,1)-lambda);
    w(ii)=1/N*sum(lambda.*Tem);
    B_array(:,ii)=Tem;
end

over_w00=1/N*sum(lambda.*lambda)-w0*w0;
for ii=1:IterNum
    over_wt0(1,ii)=1/N*sum(lambda.*lambda.*B_array(:,ii))-w0*w(ii);
end

for ii=1:IterNum
    for jj=1:IterNum
        over_w(ii,jj)=1/N*sum(lambda.*lambda.*B_array(:,ii+jj))-w(ii)*w(jj);
    end
end

end