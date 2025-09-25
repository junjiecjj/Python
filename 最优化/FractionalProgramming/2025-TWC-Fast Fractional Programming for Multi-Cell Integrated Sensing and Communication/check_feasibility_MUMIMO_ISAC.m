function [flag,P_sum] = check_feasibility_MUMIMO_ISAC(W,Pmax)
% flag 1 feasible 0 infeasible
[~,~,Q] = size(W); P_sum = 0; flag = 1;
for q = 1:Q
    P_sum = P_sum+norm(W(:,:,q),'fro')^2;
end
if P_sum>Pmax
    flag = 0;
end
end