function [iter_obj,iter_rate,iter_theta,time] = CFP_beamforming(max_iter,sigma_c,sigma_r,Nt,Nr,L,Q,H,W,T,alpha,M,K,A,d_G,mu,beta1,Pmax,eps)
% Nt: num of transmit antenna
% Nr: num of receive antenna
% M:  num of receive antenna of users
% L: num of BS
% Q: NumUser in each cell
% H: channel (M,Nt,Q*L,L)
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
            D(:,:,q,l) = (noise)\HW(:,:,q,l,q,l);
        end
    end
    %% update the auxiliary variable \Psi (Ps)  for sensing
    QQ = zeros(Nr,Nr,L);
    for l = 1:L
        QQ(:,:,l) = sigma_r*eye(Nr);
        for i = 1:L %i (l')
            if i~=l
                for j = 1:Q % A(:,:,l,i) 
                    AW = A(:,:,l,i)*W(:,:,j,i);
                    QQ(:,:,l) = AW*AW'+QQ(:,:,l);
                end
            end
        end
    end

    Ps = zeros(Nr,K,Q,L);
    for l = 1:L
        for q = 1:Q
            Ps(:,:,q,l) = (QQ(:,:,l))\(d_G(:,:,l)*W(:,:,q,l));
        end
    end

    %% update W
    DD = zeros(Nt,Nt,L);  L_Com = zeros(Nt,K,Q,L);
    for l = 1:L
        for q = 1:Q
            L_Com(:,:,q,l) = mu(q,l)*H(:,:,(l-1)*Q+q,l)'*D(:,:,q,l)*(eye(K,K)+Ga(:,:,q,l))...
                +2*T*norm(alpha(l))^2*beta1*d_G(:,:,l)'*Ps(:,:,q,l);
        end
    end
    for l = 1:L
        for i = 1:L
            for j = 1:Q
                HD = H(:,:,(i-1)*Q+j,l)'*D(:,:,j,i); % H'D
                DD(:,:,l) = mu(j,l)*HD*(eye(K)+Ga(:,:,j,i))*HD'+DD(:,:,l);
            end
            if i~=l % sensing 
                for j = 1:Q
                    APs = A(:,:,i,l)'*Ps(:,:,j,i);
                    DD(:,:,l) =DD(:,:,l)+2*T*beta1*norm(alpha(i))^2*(APs*APs');
                end
            end
        end
    end
    % find eta
    [eta] = search_eta_MUMIMO_ISAC(L_Com,DD,Pmax,eps);
    % update W
    for l = 1:L
        for q = 1:Q
            W(:,:,q,l) = (eta(l)*eye(Nt)+DD(:,:,l))\L_Com(:,:,q,l);
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

