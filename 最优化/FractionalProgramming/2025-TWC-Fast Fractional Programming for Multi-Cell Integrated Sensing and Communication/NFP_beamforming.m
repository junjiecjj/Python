function [iter_obj,iter_rate,iter_theta,time] = NFP_beamforming(max_iter,sigma_c,sigma_r,Nt,Nr,L,Q,H,W,T,alpha,M,K,A,d_G,mu,beta1,Pmax,eps)
% Nt: num of transmit antenna
% Nr: num of receive antenna
% M:  num of receive antenna of users
% L: num of BS
% Q: NumUser in each cell
% H: channel
% d_G: partial derivate of the response matrix (Nr,Nr,L)
% A: interference matrix (Nr,Nt,L,L)
% W: beamforming matrix (Nt,K,Q,L)
% mu: weight of communication
% alpha: pathloos
% T: length of frame
% beta1: weight of FI(DoA)
% K: num of data streams
iter_obj = zeros(max_iter+1,1); iter_rate = zeros(max_iter+1,1); iter_theta = zeros(max_iter+1,1);
time  = zeros(max_iter,1);
for iter = 1:max_iter
    [iter_obj(iter),iter_rate(iter), iter_theta(iter),Ga] = Compute_MU_MIMO_ISAC_obj(sigma_c,sigma_r,H,d_G,A,W,M,K,Nr,L,Q,T,alpha,mu,beta1);
    %% update the auxiliary variable (D) for communication
    tic;
    D = zeros(M,K,Q,L); HW = zeros(M,K,Q,L,Q,L);
    for l = 1:L
        for q = 1:Q
            noise = sigma_c*eye(M);
            for i = 1:L
                for j = 1:Q
                    HW(:,:,q,l,j,i) = H(:,:,(l-1)*Q+q,i)*W(:,:,j,i);
                    noise = noise+HW(:,:,q,l,j,i)*HW(:,:,q,l,j,i)';
                end
            end
            D(:,:,q,l) = (noise)\HW(:,:,q,l,q,l); % 计算用户(q,l)对应的QT的辅助变量
        end
    end
    %% update the auxiliary variable \Psi (Ps)  for sensing
    QQ = zeros(Nr,Nr,L); AW = zeros(Nr,K,L,Q,L);
    for l = 1:L
        QQ(:,:,l) = sigma_r*eye(Nr);
        for i = 1:L %i (l')
            if i~=l
                for j = 1:Q 
                    AW(:,:,l,j,i) = A(:,:,l,i)*W(:,:,j,i); % A 可以看作是干扰的信道
                    QQ(:,:,l) = AW(:,:,l,j,i)*AW(:,:,l,j,i)'+QQ(:,:,l);
                end
            end
        end
    end
    %% if iter=1, initialize Y
    if iter==1
       Ps = zeros(Nr,K,Q,L);
    for l = 1:L
        for q = 1:Q
            Ps(:,:,q,l) = (QQ(:,:,l))\(d_G(:,:,l)*W(:,:,q,l));
        end
    end
    else
        %% else use Gradient
        Y_linear = zeros(Nr,K,Q,L); % 二次项是QQ
        lambdaY = zeros(L,1);
        for l = 1:L
            lambdaY(l) = real(eigs(QQ(:,:,l),1));
            for q = 1:Q
                Y_linear(:,:,q,l) = d_G(:,:,l)*W(:,:,q,l); 
                Ps(:,:,q,l) =Ps(:,:,q,l)+1/(lambdaY(l))*(Y_linear(:,:,q,l)-QQ(:,:,l)*Ps(:,:,q,l));
            end
        end
    end
    %% update W
    DD = zeros(Nt,Nt,Q,L);  L_Com = zeros(Nt,K,Q,L);
    for l = 1:L
        for q = 1:Q
            L_Com(:,:,q,l) = mu(q,l)*H(:,:,(l-1)*Q+q,l)'*D(:,:,q,l)*(eye(K,K)+Ga(:,:,q,l))...
                +L_Com(:,:,q,l)+2*T*norm(alpha(l))^2*beta1*d_G(:,:,l)'*Ps(:,:,q,l);
        end
    end
    lambda = zeros(L,1); P = zeros(L,1);
    for l = 1:L
        for i = 1:L
            for j = 1:Q
                HD = H(:,:,(i-1)*Q+j,l)'*D(:,:,j,i); % H'D
                DD(:,:,l) = mu(q,l)*HD*(eye(K)+Ga(:,:,j,i))*HD'+DD(:,:,l);
            end
            if i~=l
                for j = 1:Q
                    APs = A(:,:,i,l)'*Ps(:,:,j,i);
                    DD(:,:,l) =DD(:,:,l)+2*T*beta1*norm(alpha(i))^2*(APs*APs');
                end
            end
        end
        % update W
        lambda(l) =real(eigs(DD(:,:,l),1));
    end
    for l = 1:L
        for q = 1:Q
            W(:,:,q,l) = W(:,:,q,l)+1/(lambda(l))*(L_Com(:,:,q,l)-DD(:,:,l)*W(:,:,q,l));
            P(l) = P(l)+norm(W(:,:,q,l),'fro')^2;
        end
        if P(l)>Pmax
            for q = 1:Q
                W(:,:,q,l) = sqrt(Pmax/P(l))*W(:,:,q,l);
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
[iter_obj(iter+1),iter_rate(iter+1), iter_theta(iter+1),~] = Compute_MU_MIMO_ISAC_obj(sigma_c,sigma_r,H,d_G,A,W,M,K,Nr,L,Q,T,alpha,mu,beta1);
end

