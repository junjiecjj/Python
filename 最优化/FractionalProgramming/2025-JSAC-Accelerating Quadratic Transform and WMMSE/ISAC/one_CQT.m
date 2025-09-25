function [iter_results,time,FI,SINR1,SINR2] = one_CQT(max_iter,sigma1,sigma2,H11,H12,H21,H22,G,alpha,Nr,M,N,w,stoptime,s_r,pp,A,v1,v2)
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明

% w1 ; w2; % weight of users' SINR
P = pp;
iter_results = zeros(max_iter+1,1); time = zeros(max_iter,1);
w1 = w; w2 = w;  sigma_s = s_r;

%% initialize v1 v2
for iter = 1:max_iter
    [iter_results(iter),~,~,~] = Compute_obj_JSAC(v1,v2,H11,H22,sigma1,sigma2,H12,H21,Nr,G,sigma_s,alpha,A,w1,w2);
    
    tic;
    %% update Y
    ys = (sigma_s*eye(Nr)+G*v2*(v2')*(G'))\A*v1;
    y1 = (sigma1*eye(N)+H12*v2*(v2')*(H12'))\H11*v1;
    y2 = (sigma2*eye(N)+H21*v2*(v2')*(H21'))\H22*v2;

    %% update v
    D1 = w2*(H21')*y2*(y2')*H21;
    D2 = alpha*(G')*ys*(ys')*G+w1*(H12'*y1*(y1')*H12);
    % search for eta
    [eta1,eta2] = find_eta(A,alpha,ys,w1,y1,H11,w2,y2,H22,D1,D2,P(1),M);
    v1 = (eta1*eye(M)+D1)\(alpha*A'*ys+w1*(H11')*y1);
    v2 = (eta2*eye(M)+D2)\(w2*(H22')*y2);
    t_run = toc;
    if iter>1
        time(iter) = time(iter-1)+t_run;
    else
        time(iter) = t_run;
    end
end
[iter_results(iter+1),FI,SINR1,SINR2] = Compute_obj_JSAC(v1,v2,H11,H22,sigma1,sigma2,H12,H21,Nr,G,sigma_s,alpha,A,w1,w2);