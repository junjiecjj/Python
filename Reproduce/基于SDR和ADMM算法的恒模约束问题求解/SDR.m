

clc;
clear all;
close all;


s=rng(1);%rand seed
N=10;
GR_num=1e3;% number, should be large enough
A = rand(N)+1i*rand(N);
A = (A*A'); % semidefinite matrix

% SDR
cvx_begin
    variable X(N,N) hermitian semidefinite
    minimize (real(trace(A*X)))
    subject to
        diag(X)==ones(N,1);
cvx_end
rank(X)
% gassian random
obj = zeros(GR_num,1);
v=zeros(N,GR_num);
[V1,D1]=eig(X);
for ii=1:GR_num
    v(:,ii) = V1*D1^(1/2)*sqrt(1/2)*(randn(N,1) + 1i*randn(N,1));
    v(:,ii) = exp(1i*angle(v(:,ii)));% guarantee constant modulus
    obj(ii)=real(trace(A*(v(:,ii)*v(:,ii)')));
    
end
[~,idx] = min(obj);
v_opt = v(:,idx);
% check constant modulus
abs(v_opt)
% check optimal value
real(trace(A*(v_opt *v_opt')))
real(trace(A*X))

