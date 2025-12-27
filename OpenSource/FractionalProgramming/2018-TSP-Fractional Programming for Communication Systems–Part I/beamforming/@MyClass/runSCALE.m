function [ instRate ] = runSCALE( obj, weight, numIter, V )

L = obj.numBS;
T = obj.numTone;
noise = obj.noise;
M = obj.numTxAnte;
H = obj.chn;
maxPower = obj.maxPower;

global convergeSCALE
convergeSCALE = nan(numIter,1);

schedule = scheduleDL( obj, weight, V );
alpha = ones(T,L); % initial alpha
lambda = zeros(1,L);

for iter = 1:numIter
    fprintf('SCALE: %d\n', iter)
    [ instRate, sinr ] = computeInstRate( obj, schedule, V ); 
    convergeSCALE(iter) = sum(weight.*instRate);  
%     if isnan(sum(sum(alpha.*log(sinr))))
%     end
    
%     cvx_begin gp
%         variables p(T,L) t(T,L);
%         maximize ( sum(sum(alpha.*log(t))) )%( sum_log(t) )%
%         subject to
%             p >= zeros(T,L);
%             for j = 1:L
%                 for z = 1:T
%                     i = schedule(z,j,1);
% %                     A = abs(squeeze(H(:,:,z,i,j)))^2;
% %                     B = abs(squeeze(H(:,:,z,i,1:L~=j)))';
%                     abs(squeeze(H(:,:,z,i,j)))^2*p(z,j)>=t(z+T*(j-1))*(sum(abs(squeeze(H(:,:,z,i,1:L~=j)))'.^2.*p(z,1:L~=j))+noise);
%                 end
%                 sum(p(:,j))<=maxPower(j);
%             end
%     cvx_end 
%     
%     sum(sum(alpha.*log(t)))
%     
%     if 1%sum(tempRate)>sum(instRate)
%         for j = 1:L
%             for z = 1:T
%                 V(:,z,j,1) = sqrt(p(z,j));
%             end
%         end
%     end

%     for in_iter = 1:1000
%     while 1
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
                VV(:,z,j) = sqrt(A/B);%min(sqrt(maxPower(j)),sqrt(A/B));
            end
%             lambda(j) = lambda(j) + .1*(sum(square_abs(VV(:,:,j)))-maxPower(j));
%             lambda(j) = max(0,lambda(j));
        end
        V = VV;        
        [ instRate, sinr ] = computeInstRate( obj, schedule, V );
    
    for j = 1:L
        A = nan(M,T,M);
        B = nan(M,M,T,M);
        for z = 1:T
            i = schedule(z,j,1);
            A(:,z) = weight(i)*alpha(z,j);
            B(:,:,z) = zeros(M,M);
            for n = 1:L
                if n==j
                    continue
                end
                m = schedule(z,n,1);
                B(:,:,z,1) = B(:,:,z,1) + sinr(z,n)/abs(H(:,:,z,m,n)*V(:,z,n))^2*abs(H(:,:,z,m,j))^2*weight(m)*alpha(z,n);
            end
            if isnan(B(:,:,z,1))
                B(:,:,z,1) = 0;
            end
        end  
        for z = 1:T
            V(:,z,j,1) = sqrt(A(:,z,1)/B(:,:,z));
            if isnan(V(:,z,j,1))
                V(:,z,j,1) = 0;
            end
        end        
        if sum(square_abs(V(:,:,j)))<=maxPower(j)
            continue
        end        
        muLeft = 0;
        muRight = 1;
        while 1
            for z = 1:T
                V(:,z,j,1) = sqrt(A(:,z,1)/(muRight+B(:,:,z)));
            end
            if sum(square_abs(V(:,:,j)))<=maxPower(j)
                break
            end
            muRight = muRight*10;
        end
        while 1
            mu = (muLeft+muRight)/2;
            for z = 1:T
                V(:,z,j,1) = sqrt(A(:,z,1)/(mu+B(:,:,z)));
            end            
            if abs(sum(square_abs(V(:,:,j)))-maxPower(j)) < maxPower(j)/1e8
                break
            end

            if  sum(square_abs(V(:,:,j))) > maxPower(j)
                muLeft = mu;
            else
                muRight = mu;
            end
        end
    end
       
    [ instRate, sinr ] = computeInstRate( obj, schedule, V );
    
    % update alpha
    if 1%mod(iter,5)==0
        alpha = sinr./(1+sinr);
    end
end

end