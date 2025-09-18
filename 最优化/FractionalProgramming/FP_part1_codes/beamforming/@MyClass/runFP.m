function [ instRate ] = runFP( obj, weight, numIter, V )

L = obj.numBS;
K = obj.numUser;
M = obj.numTxAnte;
N = obj.numRxAnte;
T = obj.numTone;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;

global convergeFP
convergeFP = nan(numIter,1);

schedule = scheduleDL( obj, weight, V );

% cvx_begin gp
%     variables p(T,L) t(T,L);
%     maximize ( sum(sum(log(t))) )%( sum_log(t) )%
%     subject to
%         p >= zeros(T,L);
%         for j = 1:L
%             for z = 1:T
%                 i = schedule(z,j,1);
% %                     A = abs(squeeze(H(:,:,z,i,j)))^2;
% %                     B = abs(squeeze(H(:,:,z,i,1:L~=j)))';
%                 abs(squeeze(H(:,:,z,i,j)))^2*p(z,j)>=t(z+T*(j-1))*(sum(abs(squeeze(H(:,:,z,i,1:L~=j)))'.^2.*p(z,1:L~=j))+noise);
%             end
%             sum(p(:,j))<=maxPower(j);
%         end
% cvx_end 
% if 1%sum(tempRate)>sum(instRate)
%     for j = 1:L
%         for z = 1:T
%             V(:,z,j,1) = sqrt(p(z,j));
%         end
%     end
% end

for iter = 1:numIter
    fprintf('FP: %d\n', iter)
    [ instRate, sinr ] = computeInstRate( obj, schedule, V );
    convergeFP(iter) = sum(weight.*instRate);
    
    [ y ] = updateY( L, T, M, N, noise, H, schedule, V );
    
%     VV = V;
%     for j = 1:L
%         for z = 1:T
%             i = schedule(z,j,1);
%             A = abs(y(1,:,z,j)*H(:,:,z,i,j)/(1+sinr(z,j)));
%             B = 0;
%             for n = 1:L
%                 if n == j
%                     continue
%                 end
%                 m = schedule(z,n,1);
%                 B = B + abs(y(1,:,z,n)*H(:,:,z,m,j))^2/(1+sinr(z,n));
%             end
%             VV(:,z,j) = min(A/B,maxPower(j));
%         end
%     end
%     
%     V = VV;\\
    
    A = zeros(L*M,M,T,L,M);
    for j = 1:L
        for s = 1:M
            for z = 1:T
                i = schedule(z,j,s);
                for n = 1:L
                    for t = 1:M
                        if n==j && t==s
                            continue
                        end
                        A(M*(n-1)+t,:,z,j,s) = y(1,:,z,j,s)*H(:,:,z,i,n);
                    end
                end
            end
        end
    end
    
    cvx_begin
        variable F(M,L*M,T) complex 
        variable theta(L*M,T)
        maximize ( sum(sum(log(theta))) )%( sum(sum(theta)) )%( W*log(theta)' ) %geo_mean(theta)%
        subject to
            for j = 1:L
                sum(sum_square_abs(F(:,j,:))) <= maxPower(j) % !!! use this when numTxAnte==1 and numTone==1
%                 sum(sum(sum(square_abs(F(:,M*(j-1)+1:M*(j-1)+M,:))))) <= maxPower(j)
                for s = 1:M
                    for z = 1:T
                        1 + 2*real(y(1,:,z,j,s)*H(:,:,z,schedule(z,j,s),j)*F(:,M*(j-1)+s,z)) - sum_square_abs(y(1,:,z,j,s))*noise ...
                            - sum_square_abs(diag(A(:,:,z,j,s)*F(:,:,z))) >= theta(M*(j-1)+s,z)%exp(theta(M*(j-1)+s,z))
                    end
                end
            end
    cvx_end
    
    V = nan(M,T,L,M);
    for j = 1:L
        for s = 1:M
            for z = 1:T
                V(:,z,j,s) = F(:,M*(j-1)+s,z);
            end
        end
    end
    
%     %%
%     [ instRate, sinr ] = computeInstRate( obj, schedule, V );
%     grad = nan(1,T,L);
%     for j = 1:L
%         for z = 1:T
%             i = schedule(z,j,1);
%             grad(1,z,j) = abs(y(1,:,z,j)*H(:,:,z,i,j)/V(:,z,j))/(1+sinr(z,j));
%             for n = 1:L
%                 if n==j
%                     continue
%                 end
%                 m = schedule(z,n,1);
%                 grad(1,z,j) = grad(1,z,j) - abs(y(1,:,z,n)*H(:,:,z,m,j))^2/(1+sinr(z,n));
%             end
%         end
%     end
%     
%     [ instRate, sinr ] = computeInstRate( obj, schedule, V );
%     SWR = sum(instRate);
%     step = 1;
%     while 1
%         p_new = max(0, V.^2+step*grad);
%         [ instRate, ~ ] = computeInstRate( obj, schedule, sqrt(p_new) );
%         SWR_new = sum(instRate);
%         
%         if SWR_new>=SWR || abs(SWR_new-SWR)<1e-2
%             V = sqrt(p_new);
%             break
%         else
%             step = step/2;
%         end
%     end
   
end

end    

function [ y ] = updateY( L, T, M, N, noise, H, schedule, V )

y = nan(1,N,T,L,M);
for j = 1:L
    for s = 1:M
        for z = 1:T
            i = schedule(z,j,s);
            B = eye(N)*noise;
            for n = 1:L
                for t = 1:M
                    if n==j && t==s
                        continue
                    end
                    B = B + H(:,:,z,i,n)*V(:,z,n,t)*V(:,z,n,t)'*H(:,:,z,i,n)';
                end
            end
            
            y(1,:,z,j,s) = V(:,z,j,s)'*H(:,:,z,i,j)'/B;
        end
    end
end

end