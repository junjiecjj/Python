function [ instRate ] = runSCALE( obj, weight, numIter, V )

L = obj.numBS;
T = obj.numTone;
noise = obj.noise;
H = obj.chn;
maxPower = obj.maxPower;

global convergeSCALE
convergeSCALE = nan(numIter,1);

schedule = scheduleDL( obj, weight, V );
alpha = ones(T,L); % initial alpha

for iter = 1:numIter
    fprintf('SCALE: %d\n', iter)
    [ instRate, sinr ] = computeInstRate( obj, schedule, V );
    convergeSCALE(iter) = sum(weight.*instRate);   
    
    for inn_iter = 1:10
        VV = V;
        for j = 1:L
            for z = 1:T
                i = schedule(z,j,1); % M = 1
                A = weight(i)*alpha(z,j);
                B = 0;
                for n = 1:L
                    if n == j
                        continue
                    end
                    m = schedule(z,n,1);
                    B = B + sinr(z,n)/abs(H(:,:,z,m,n)*V(:,z,n))^2*abs(H(:,:,z,m,j))^2*weight(m)*alpha(z,n);
                end
                VV(:,z,j) = sqrt(min(A/B,maxPower(j)));
            end
        end
        V = VV;
    end
%     cvx_begin gp
%         variables p(T,L) t(T*L,1);
%         maximize ( sum_log(t) )%( sum(sum(alpha.*log(t))) )
%         subject to
%             p >= zeros(T,L);
%             for j = 1:L
%                 for z = 1:T
%                     i = schedule(z,j,1);
%                     A = abs(squeeze(H(:,:,z,i,j)))^2;
%                     B = abs(squeeze(H(:,:,z,i,1:L~=j)))';
%                     A*p(z,j)>=t(z+T*(j-1))^alpha(z,j)*(sum(B.^2.*p(z,1:L~=j))+noise);
%                 end
%                 sum(p(:,j))<=maxPower(j);
%             end
%     cvx_end 
    
%     [ tempRate, ~ ] = computeInstRate( obj, schedule, sqrt(p) );
    
%     if sum(tempRate)>sum(instRate)
%         for j = 1:L
%             for z = 1:T
%                 V(:,z,j,1) = sqrt(p(z,j));
%             end
%         end
%     end
    
    [ instRate, sinr ] = computeInstRate( obj, schedule, V );
    % update alpha
    alpha = sinr./(1+sinr);
end

end