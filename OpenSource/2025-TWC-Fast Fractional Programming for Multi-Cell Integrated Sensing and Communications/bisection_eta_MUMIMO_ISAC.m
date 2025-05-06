function [fin_eta]= bisection_eta_MUMIMO_ISAC(l,L_Com,DD,eps,Pmax)
% 函数内的W对应的是W(:,:,:,l)
%   此处显示详细说明
eta = 2;
%% find ub (lb<=eta<=ub)
[W] = update_W_MUMIMO_ISAC(l,L_Com,DD,eta); % W(:,:,:,l)
[flag,~] = check_feasibility_MUMIMO_ISAC(W,Pmax);
while flag~=1 % flag = 1, feasible
    eta=eta^2;% infeasible, 说明eta不够大
    [W] = update_W_MUMIMO_ISAC(l,L_Com,DD,eta);
    [flag,~] = check_feasibility_MUMIMO_ISAC(W,Pmax);
end
%% find lb (lb<=eta<=ub)
if eta == 2 
    lb = 0;
else
    lb = sqrt(eta);
end
ub = eta;
%% 
[W] = update_W_MUMIMO_ISAC(l,L_Com,DD,eta);
[~,p_W] = check_feasibility_MUMIMO_ISAC(W,Pmax);
iter = 0;
%% when (Pmax-p(ub))/Pmax>eps
while (Pmax-p_W)/Pmax>eps 
    if iter>100 && Pmax>p_W
        break
    end
    iter = iter+1;
    mid = (lb+ub)/2;
    % if ub==0
    %     1
    % end
    eta = mid; % 通过检验中间值的目标值重新划分区间
    [W] = update_W_MUMIMO_ISAC(l,L_Com,DD,eta);
    [flag,~] = check_feasibility_MUMIMO_ISAC(W,Pmax);
    if flag==0
        lb = mid;
    else
        ub = mid;
    end
    eta = ub;
    % 更新u的值
    [W] = update_W_MUMIMO_ISAC(l,L_Com,DD,eta);
    [~,p_W] = check_feasibility_MUMIMO_ISAC(W,Pmax);
end
fprintf('needs %d iter to find eta(%d)\n',iter,l);
fin_eta = ub;
end