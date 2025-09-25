function [iter_results,time] = EQT(max_iter,sigma,M,N,L,Q,chn,w,Pmax,V)
% t: auxiliary variable SINR (NumUser NumCell)
% Y: auxiliary variable (Rx NumUser NumCell)
% L: NumCell
% Q: NumUser in each cell
% chn: channel (Rx Tx (NumUser x NumCell) NumCell)
% w: weight of each rate (NumUser NumCell)
% V: precoding vectors (Tx NumUser NumCell)
% M Tx N Rx
iter_results = zeros(max_iter+1,1); time = zeros(max_iter,1);
ex_X = zeros(N,Q,L,Q,L); S = zeros(M,L,Q,L);
Y = zeros(N,Q,L);
prev_V = V;
for iter = 1:max_iter
    [iter_results(iter),t] = Compute_MIMO_obj(sigma,M,N,L,Q,chn,w,V);
    tic
    beta = (iter-1)/(iter+2);
    ex_V = V + beta*(V-prev_V);
    %% update Y using ex_V
    for l = 1:L
        for q = 1:Q
            noise = sigma*eye(N);
            for i = 1:L
                for j = 1:Q
                    ex_X(:,q,l,j,i) = chn(:,:,(l-1)*3+q,i)*ex_V(:,j,i); 
                    noise = noise+ex_X(:,q,l,j,i)*ex_X(:,q,l,j,i)';
                end
            end
            Y(:,q,l) = (noise)\ex_X(:,q,l,q,l);
        end
    end
    T = ex_V;
    prev_V = V;
    %% Compute D
    D = zeros(M,M,L); lambda = zeros(L,1);
    for l = 1:L
        for i = 1:L
            for j = 1:Q
                S(:,i,j,l) = chn(:,:,(i-1)*3+j,l)'*Y(:,j,i);
                D(:,:,l) = D(:,:,l)+w(j,i)*(1+t(j,i))*S(:,i,j,l)*S(:,i,j,l)';
            end
        end
         lambda(l) = (real(eigs(D(:,:,l),1)));
    end
   
    %% update V
    for l = 1:L
        for q = 1:Q
            V(:,q,l) = T(:,q,l)+1/lambda(l)*(w(q,l)*(1+t(q,l))*S(:,l,q,l)-D(:,:,l)*T(:,q,l));
        end
    end
   
    % Projection
    [flags,sns] = check_MIMO_SNS(V,Pmax);
    while sum(flags)~=7
        for l = 1:L
            if flags(l)==0
                V(:,:,l) = V(:,:,l)*sqrt(Pmax/sns(l));
            end
        end
        [flags,sns] = check_MIMO_SNS(V,Pmax);
    end
    t_run = toc;
    if iter>1
    time(iter) = time(iter-1)+t_run;
    else
        time(iter) = t_run;
    end
   
end
[iter_results(iter+1),~] = Compute_MIMO_obj(sigma,M,N,L,Q,chn,w,V);