function [ instRate ] = runNewton( obj, weight, numIter, V )

% this method only applies to SISO for power control

L = obj.numBS;
T = obj.numTone;
H = obj.chn;
maxPower = obj.maxPower;

p = square_abs(V);
global convergeNewton
convergeNewton = nan(numIter,1);

schedule = scheduleDL( obj, weight, V );

for iter = 1:numIter
    fprintf('Newton: %d\n', iter)
    
    [ instRate, sinr ] = computeInstRate( obj, schedule, sqrt(p) );
    convergeNewton(iter) = sum(instRate);
    
    delt_p = zeros(1,T,L);
    for j = 1:L
        for z = 1:T
            if p(:,z,j)==0
                continue
            end
            i = schedule(z,j,1);
            A = weight(i)/p(:,z,j)/(1+1/sinr(z,j));
            for n = 1:L
                if n==j
                    continue
                end
                if p(:,z,n)==0
                    continue
                end
                m = schedule(z,n,1);
                A = A - weight(m)*abs(H(:,:,z,m,j))^2*sinr(z,n)^2/p(:,z,n)/abs(H(:,:,z,m,n))^2/(1+sinr(z,n));
            end
            
            B = weight(i)/p(:,z,j)^2/(1+1/sinr(z,j))^2;
            delt_p(:,z,j) = A/B;
        end
    end
    
    % backtracking line search
    swr = sum(instRate.*weight); % sum weighted rate
    
    step = 1;
    while 1
        p_new = max(0, p+step*delt_p);
        
        for j = 1:L % projection
            sum_p = sum(p_new(:,:,j));
            if sum_p > maxPower(j)
               cvx_begin
                    variable x(1,T)
                    minimize norm(x-p_new(1,:,j))
                    subject to
                        x >= zeros(1,T)
                        sum(x) <= maxPower(j)
               cvx_end
               p_new(1,:,j) = x;
            end
            
        end
        
        instRate_new = computeInstRate( obj, schedule, sqrt(p_new) );
        swr_new = sum(instRate_new.*weight);
        
        swr_new - swr
        if swr_new >= swr || abs(swr_new-swr)<1e-1
            p = p_new;
            break
        else
            step = step/2;
        end
    end
end  

end    

% u = 0:.1:10;
% y = (1e9)*log(u);
% plot(u,y);