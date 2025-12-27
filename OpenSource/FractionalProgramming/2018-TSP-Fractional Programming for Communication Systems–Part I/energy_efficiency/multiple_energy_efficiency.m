clc; clear
cvx_solver SDPT3

numIter = 50; % num of iter
utility_dinkelbach = zeros(numIter,1);
utility_proposed = zeros(numIter,1);

K = 3; % number of users
M = K; % ... tx antennas
N = 2; % ... rx ...
F = 1;
maxPower = 10^((21-30)/10)*F;
noise =  10^((-100-30)/10)*F;
po = 10^((5-30)/10)*F;
chn = 1e-12; % pathloss
H = nan(N,M,K);
for i = 1:K
    H(:,:,i) = (sqrt(chn)*randn(N,M) + 1i*sqrt(chn)*randn(N,M))/sqrt(2);
end
% H = sqrt(chn)*randn(N,M,K) + 1i*sqrt(chn)*randn(N,M,K);
numIter = 21;
% initV = ones(M,K)*sqrt((maxPower-po)/M/M);
initV = (rand(M,K)+1i*rand(M,K))*sqrt((maxPower-po)/M/M/2);

%% Dinkelbach-Quadratic
efficiencyDQ = nan(numIter,1);
y = nan(N,K);
V = initV;
for iter = 1:numIter
    % update y
    for i = 1:K
        A = H(:,:,i)*V(:,i);
        B = noise*eye(N);
        for j = 1:K
            if j==i
                continue
            end
            B = B + H(:,:,i)*V(:,j)*V(:,j)'*H(:,:,i)';
        end
        y(:,i) = B\A;
    end
    % update z
    fq = 0;
    for i = 1:K
        A = 2*real(y(:,i)'*H(:,:,i)*V(:,i));
        B = noise*eye(N);
        for j = 1:K
            if j==i
                continue
            end
            B = B + H(:,:,i)*V(:,j)*V(:,j)'*H(:,:,i)';
        end
        fq = fq + log(1+A-abs(y(:,i)'*B*y(:,i)));
    end
    z = fq/(sum(sum(abs(V).^2))+po);
    efficiencyDQ(iter) = z;
%     %
%     rate = 0;
%     for i = 1:K
%         B = noise*eye(N);
%         for j = 1:K
%             if j==i
%                 continue
%             end
%             B = B + H(:,:,i)*V(:,j)*V(:,j)'*H(:,:,i)';
%         end
%         rate = rate + log(1+abs(V(:,i)'*H(:,:,i)'*inv(B)*H(:,:,i)*V(:,i)));
%     end
    
    A = zeros(K,K,K);
    for i = 1:K
        for j = 1:K
            if j~=i
                A(j,:,i) = y(:,i)'*H(:,:,i);
            end
        end
    end
    
%     rate = 0;
%     for i = 1:K
%         rate = rate + log(1 + 2*real(y(:,i)'*H(:,:,i)*V(:,i)) - sum_square_abs(y(:,i))*noise...
%                     - sum_square_abs(diag(A(:,:,i)*V)));
%     end
    theta = nan(K,1);
    for i = 1:K
        theta = log(1 + 2*real(y(:,i)'*H(:,:,i)*V(:,i)) - sum_square_abs(y(:,i))*noise...
            - sum_square_abs(A(:,:,i)*V));
    end
    sum(theta) - z*sum(sum(square_abs(V)))
    
    cvx_begin
        variable V(M,K) complex
        variable theta(K,1)
        maximize( sum(theta) - z*sum(sum(square_abs(V))) )
        subject to
            sum(sum(square_abs(V))) <= maxPower-po
            for i = 1:K
                1 + 2*real(y(:,i)'*H(:,:,i)*V(:,i)) - sum_square_abs(y(:,i))*noise...
                    - sum_square_abs(diag(A(:,:,i)*V)) >= exp(theta(i))
            end
    cvx_end
end

%% Double-Quadratic
efficiencyQQ = nan(numIter,1);
y = nan(N,K);
V = initV;
for iter = 1:numIter
    % update y
    for i = 1:K
        A = H(:,:,i)*V(:,i);
        B = noise*eye(N);
        for j = 1:K
            if j==i
                continue
            end
            B = B + H(:,:,i)*V(:,j)*V(:,j)'*H(:,:,i)';
        end
        y(:,i) = B\A;
    end
    % update z
    fq = 0;
    for i = 1:K
        A = 2*real(y(:,i)'*H(:,:,i)*V(:,i));
        B = noise*eye(N);
        for j = 1:K
            if j==i
                continue
            end
            B = B + H(:,:,i)*V(:,j)*V(:,j)'*H(:,:,i)';
        end
        fq = fq + log(1+A-abs(y(:,i)'*B*y(:,i)));
    end
    
    
    z = sqrt(fq)/(sum(sum(abs(V).^2))+po);
    efficiencyQQ(iter) = fq/(sum(sum(abs(V).^2))+po);
    
    A = zeros(K,K,K);
    for i = 1:K
        for j = 1:K
            if j~=i
                A(j,:,i) = y(:,i)'*H(:,:,i);
            end
        end
    end
    
%     theta = nan(K,1);
%     for i = 1:K
%         theta(i) = log(1 + 2*real(y(:,i)'*H(:,:,i)*V(:,i)) - sum_square_abs(y(:,i))*noise...
%             - sum_square_abs(diag(A(:,:,i)*V)));
%     end
%     efficiencyQQ(iter)
%     2*z*sqrt(sum(theta)) - z^2*(sum(sum(square_abs(V)))+po)
         
    cvx_begin
        variable V(M,K) complex
        variable theta(K,1)
%         maximize( 2*z*sqrt(3*log(geo_mean(theta))) - z^2*sum(sum(square_abs(V))) )
        maximize( 2*z*sqrt(sum(theta)) - z^2*sum(sum(square_abs(V))) )
        subject to
            sum(sum(square_abs(V))) <= maxPower-po
            for i = 1:K
                1 + 2*real(y(:,i)'*H(:,:,i)*V(:,i)) - sum_square_abs(y(:,i))*noise...
                    - sum_square_abs(diag(A(:,:,i)*V)) >= exp(theta(i))
            end
    cvx_end
end

hold on
plot(efficiencyDQ)
plot(efficiencyQQ,'r')

save(strrep(strrep(num2str(clock),' ',''),'.','_'), 'efficiencyDQ','efficiencyQQ');

% %% plot the function
% p = 0:maxPower/100:maxPower;
% ee = log2(1+chn*p/noise)./(p+po);
% plot(p,ee);
% 
% %% compare convergence
% ee_dinkelbach = nan(num_iter,1); % energy efficiency by Dinkelbach
% ee_quadratic = nan(num_iter,1); % energy efficiency by Quadratic
% 
% % Dinkelbach
% p = maxPower;
% for iter = 1:num_iter
%     y = log(1+chn*p/noise)/(p+po);
%     ee_dinkelbach(iter) = y/log(2);
%     
%     cvx_begin
%         variable p
%         maximize( log(1+chn*p/noise) - y*(p+po) )
%         subject to
%             p >= 0
%             p <= maxPower - po
%     cvx_end
% end
% 
% % quadratic
% p = maxPower;
% for iter = 1:num_iter
%     y = sqrt(log(1+chn*p/noise))/(p+po);
%     ee_quadratic(iter) = log2(1+chn*p/noise)/(p+po);
%     
%     cvx_begin
%         variable p
%         maximize( 2*y*sqrt(log(1+chn*p/noise)) - y^2*(p+po) )
%         subject to
%             p >= 0
%             p <= maxPower - po
%     cvx_end
% end
% 
% figure; hold on
% plot(0:num_iter-1, ee_dinkelbach,'r-o')
% plot(0:num_iter-1, ee_quadratic,'b-*')
% 
% % figure; hold on
% % plot(1:num_iter, max(ee_dinkelbach)-ee_dinkelbach,'r')
% % plot(1:num_iter, max(ee_quadratic)-ee_quadratic)
% % 
% % figure; hold on
% % gap_dinkelbach = max(ee_dinkelbach)-ee_dinkelbach;
% % gap_quadratic = max(ee_quadratic)-ee_quadratic;
% % plot(0:5, gap_dinkelbach(1:6),'r')
% % plot(0:9, gap_quadratic(1:10))
