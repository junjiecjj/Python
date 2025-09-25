function [iter_results,time] = NQT(max_iter,sigma,M,N,L,Q,chn,w,Pmax,V)
% t: auxiliary variable SINR (NumUser NumCell)
% Y: auxiliary variable (Rx NumUser NumCell)
% L: NumCell
% Q: NumUser in each cell
% chn: channel (Rx Tx (NumUser x NumCell) NumCell)
% w: weight of each rate (NumUser NumCell)
% V: precoding vectors (Tx NumUser NumCell)
% M Tx N Rx
iter_results = zeros(max_iter+1,1); time = zeros(max_iter,1);
X = zeros(N,L,Q,L); S = zeros(M,L,Q,L);
Y = zeros(N,Q,L); 

for iter = 1:max_iter
    [iter_results(iter),t] = Compute_MIMO_obj(sigma,M,N,L,Q,chn,w,V);
    tic
    %% update Y
    for l = 1:L
        for q = 1:Q
            noise = sigma*eye(N);
            for i = 1:L
                for j = 1:Q
                    X(:,q,l,j,i) = chn(:,:,(l-1)*3+q,i)*V(:,j,i);
                    noise = noise+X(:,q,l,j,i)*X(:,q,l,j,i)';
                end
            end
            Y(:,q,l) = (noise)\X(:,q,l,q,l);
        end
    end
    %% Compute D
    D = zeros(M,M,L); lambda = zeros(L,1);
    for l = 1:L
        for i = 1:L
            for j = 1:Q
                S(:,i,j,l) = chn(:,:,(i-1)*3+j,l)'*Y(:,j,i);
                D(:,:,l) = D(:,:,l)+w(j,i)*(1+t(j,i))*S(:,i,j,l)*S(:,i,j,l)';
            end
        end 
        lambda(l) = eigs(D(:,:,l),1); % one can also use    lambda(l) = norm(D(:,:,l),'fro');
    end
    %% update V
    for l = 1:L
        for q = 1:Q
            V(:,q,l) = V(:,q,l)+1/lambda(l)*(w(q,l)*(1+t(q,l))*S(:,l,q,l)-D(:,:,l)*V(:,q,l));
        end
    end
    % Projection
    [flags,sns] = check_MIMO_SNS(V,Pmax);
    if sum(flags)~=7
        for l = 1:L
            if flags(l)==0
                V(:,:,l) = V(:,:,l)*sqrt(Pmax/sns(l));
            end
        end
    end
    t_run = toc;
    if iter>1
    time(iter) = time(iter-1)+t_run;
    else
        time(iter) = t_run;
    end
end
[iter_results(iter+1),~] = Compute_MIMO_obj(sigma,M,N,L,Q,chn,w,V);