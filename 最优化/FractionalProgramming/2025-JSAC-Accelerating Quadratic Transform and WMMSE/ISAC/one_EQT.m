function [iter_results,time,FI,SINR1,SINR2] = one_EQT(max_iter,sigma1,sigma2,H11,H12,H21,H22,G,alpha,Nr,M,N,w,s_r,pp,A,v1,v2)
% w1 ; w2; % weight of users' SINR
P = pp;
iter_results = zeros(max_iter+1,1); time = zeros(max_iter,1);
w1 = w; w2 = w;  sigma_s = s_r;

prev_v1 = v1; prev_v2 = v2;
for iter = 1:max_iter

    [iter_results(iter),~,~,~] = Compute_obj_JSAC(v1,v2,H11,H22,sigma1,sigma2,H12,H21,Nr,G,sigma_s,alpha,A,w1,w2);
    tic;
    beta = (iter-1)/(iter+2);
    ex_v1 = v1+beta*(v1-prev_v1);
    ex_v2 = v2+beta*(v2-prev_v2);

    %% compute Y using ex_v1, ex_v2
    ys = (sigma_s*eye(Nr)+G*ex_v2*(ex_v2')*(G'))\A*ex_v1;
    y1 = (sigma1*eye(N)+H12*ex_v2*(ex_v2')*(H12'))\H11*ex_v1;
    y2 = (sigma2*eye(N)+H21*ex_v1*(ex_v1')*(H21'))\H22*ex_v2;

    t_v1 = ex_v1; t_v2 = ex_v2; prev_v1 = v1; prev_v2 = v2;
    %% compute D
    D1 = w2*(H21')*y2*(y2')*H21;
    D1 = (D1+D1.')/2;
    D2 = alpha*(G')*ys*(ys')*G+w1*(H12'*y1*(y1')*H12);
    D2 = (D2+D2.')/2;
    % update lambda
    lambda1 = (real(eigs(D1,1)));
    lambda2 = (real(eigs(D2,1)));

    %% update v1 v2
    v1 = 1/lambda1*(alpha*A'*ys+w1*H11'*y1+(lambda1*eye(M)-D1)*t_v1); % 省略了更新z的步骤
    v2 = 1/lambda2*(w2*H22'*y2+(lambda2*eye(M)-D2)*t_v2);

    if norm(v1)^2>P(1)
        v1 = sqrt(P(1))/(norm(v1))*v1;
    end
    if norm(v2)^2>P(2)
        v2 = sqrt(P(2))/(norm(v2))*v2;
    end
    t_run = toc;
    if iter>1
        time(iter) = time(iter-1)+t_run;
    else
        time(iter) = t_run;
    end
end
[iter_results(iter+1),FI,SINR1,SINR2] = Compute_obj_JSAC(v1,v2,H11,H22,sigma1,sigma2,H12,H21,Nr,G,sigma_s,alpha,A,w1,w2);

end