%% This demo shows how to minimize the sum of priority AoI.
% This work was done by Yannan CHEN from CUHK (SZ).
% If you have any questions regarding this code, please feel free to contact me at yannanchen@link.cuhk.edu.cn.
clear; clc
iter_num = 10;         % number of max iteration
rand_index_max = 1;  % number of random test (random initialization point)
mu = 1;

All_sources = 3;
different_sources = length(All_sources);
final_results = zeros(different_sources,1);
for source_index = 1:different_sources
    sources = All_sources(source_index);
    all_results = zeros(iter_num+1,rand_index_max);
    for rand_index = 1:rand_index_max
        n = sources;
        obj = zeros(iter_num+1,1); time = zeros(iter_num,1);
        rho = ones(n,1); 
        AA = zeros(n,iter_num);
        for iter = 1:iter_num
            [obj(iter), AA(:,iter)]= sum_priority_AoI_obj(mu,rho);
            q = zeros(n,1); qq = zeros(n,1);
            % Find the optimal auxiliary variable
            tic
            for i = 1:n
                q(i) = sqrt(mu*(ones(1,i-1)*rho(1:i-1)+1))/((ones(1,i-1)*rho(1:i-1))^2+3*(ones(1,i-1)*rho(1:i-1))+1); % For the first ratio term in the AoI expression
                qq(i) = sqrt(mu*rho(i))/(((ones(1,i-1)*rho(1:i-1)+1)^2)); % For the second ratio term in the AoI expression
            end
            % Update rho
            cvx_begin
            variable rho(n) nonnegative
            expression AoI(n)
            for i = 1:n
                AoI(i) = inv_pos(2*q(i)*sqrt(mu*(ones(1,i-1)*rho(1:i-1)+1))-q(i)^2*((ones(1,i-1)*rho(1:i-1))^2+3*(ones(1,i-1)*rho(1:i-1))+1))+...
                    inv_pos(2*qq(i)*sqrt(mu*rho(i))-qq(i)^2*(((ones(1,i-1)*rho(1:i-1)+1)^2)));
            end
            minimize sum(AoI)
            subject to
            rho<=ones(n,1)
            cvx_end
        end
        t_run = toc;
        if iter>1
            time(iter) = time(iter-1)+t_run;
        else
            time(iter) = t_run;
        end
        [obj(iter+1), ~]= sum_priority_AoI_obj(mu,rho);
        all_results(:,rand_index)  = obj;
    end
    a = sum(all_results,2)/rand_index_max;
    final_results(source_index) = a(end);

end