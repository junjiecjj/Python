function [obj,weighted_rate, weighted_FI_theta,Ga] = Compute_MU_MIMO_ISAC_obj(sigma_c,sigma_r,H,d_G,A,W,M,K,Nr,L,Q,T,alpha,mu,beta1)
% Nt: num of transmit antenna
% Nr: num of receive antenna
% M:  num of receive antenna of users
% L: num of BS
% Q: NumUser in each cell
% H: channel (M,Nt,Q*L,L)
% G: response matrix (Nr,Nr,L,L)
% d_G: partial derivate of the response matrix (Nr,Nr,L)
% A: alpha*G (Nr,Nt,L,L)
% W: beamforming matrix (Nt,K,Q,L)
% mu: weight of communication (Q,L)
% alpha: xishu (L)
% T: length of frame
% beta1: weight of FI(DoA)  beta2: weight of FI
% K: num of data streams
Ga = zeros(K,K,Q,L); HW = zeros(M,K,Q,L,Q,L); Rate = zeros(Q,L);
for l = 1:L
    for q = 1:Q
        noise = sigma_c*eye(M);
        for i = 1:L
            for j = 1:Q
                HW(:,:,q,l,j,i) = H(:,:,(l-1)*Q+q,i)*W(:,:,j,i);
                noise = noise+HW(:,:,q,l,j,i)*HW(:,:,q,l,j,i)';
            end
        end
        noise = noise-HW(:,:,q,l,q,l)*HW(:,:,q,l,q,l)';
        Ga(:,:,q,l) = HW(:,:,q,l,q,l)'/(noise)*HW(:,:,q,l,q,l);
        Rate(q,l) = log2(det(eye(K)+Ga(:,:,q,l)));
    end
end
QQ = zeros(Nr,Nr,L); 
for l = 1:L
    QQ(:,:,l) = sigma_r*eye(Nr);
    for i = 1:L %i (l')
        if i~=l
            for j = 1:Q % A(:,:,l,i) should be i transmit, l receive
                AW = A(:,:,l,i)*W(:,:,j,i);
                QQ(:,:,l) = AW*AW'+QQ(:,:,l);
            end
        end
    end
end
all_FI_theta = zeros(Q,L);  FI_theta = zeros(L,1); 
for l = 1:L
    for q = 1:Q
        d_GW = d_G(:,:,l)*W(:,:,q,l);
        all_FI_theta(q,l) = trace(d_GW'/QQ(:,:,l)*d_GW);
    end
    FI_theta(l) = sum(all_FI_theta(:,l))*norm(alpha(l))^2;
end
weighted_rate = real(sum(sum(Rate.*mu))); weighted_FI_theta = real(sum(FI_theta))*2*T*beta1;
obj = weighted_rate+weighted_FI_theta;
end



