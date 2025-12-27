function [iter_results,time] = CQT(max_iter,sigma,M,N,L,Q,chn,w,Pmax,V)
% t: auxiliary variable SINR (NumUser NumCell)
% Y: auxiliary variable (Rx NumUser NumCell)
% L: NumCell
% Q: NumUser in each cell
% chn: channel (Rx Tx 1 (NumUser x NumCell) NumCell)
% w: weight of each rate (NumUser NumCell)
% V: precoding vectors (Tx NumUser NumCell)
% M Tx N Rx
iter_results = zeros(max_iter+1,1); time = zeros(max_iter,1);
X = zeros(N,Q,L,Q,L);
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

    %% find eta
    eta = search_eta(w,t,chn,Y,Pmax);
    %% update v
    [V] = update_V(eta,w,t,chn,Y);
    t_run = toc;
    if iter>1
        time(iter) = time(iter-1)+t_run;
    else
        time(iter) = t_run;
    end
end
[iter_results(iter+1),~] = Compute_MIMO_obj(sigma,M,N,L,Q,chn,w,V);