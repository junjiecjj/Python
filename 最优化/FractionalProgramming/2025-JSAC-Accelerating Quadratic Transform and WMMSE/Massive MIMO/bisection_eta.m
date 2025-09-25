function fin_eta = bisection_eta(l,eps,chn,w,t,Y,Pmax)
% l: index of cell
% eps: error tolerance
% chn: channel (Rx Tx 1 (NumUser x NumCell)) lth cell
% w:  weight of each rate (NumUser NumCell)
% t: auxiliary variable SINR (NumUser NumCell)
% Y: auxiliary variable (Rx NumUser NumCell)
[~,~,L] = size(Y);
eta = zeros(L,1); eta(l) = 2;
[one_V] = update_one_V(eta,w,t,chn,Y,l);
[flag,~] =  check_MIMO_SNS(one_V,Pmax);
while flag~=1
    eta(l)=eta(l)^2;
    [one_V] = update_one_V(eta,w,t,chn,Y,l);
    [flag,~] =  check_MIMO_SNS(one_V,Pmax);
end
if eta(l) == 2
    lb = 0;
else
    lb = sqrt(eta(l));
end
ub = eta(l);
[one_V] = update_one_V(eta,w,t,chn,Y,l);
[~,sns] =  check_MIMO_SNS(one_V,Pmax);
% try
while (Pmax-sns)/Pmax>eps
    mid = (lb+ub)/2;
    eta(l) = mid;
    one_V = update_one_V(eta,w,t,chn,Y,l);
    [flag,sns] =  check_MIMO_SNS(one_V,Pmax);
    if flag==0
        lb = mid;
    else
        ub = mid;
    end
    eta(l) = ub;
    % 更新u的值
    one_V = update_one_V(eta,w,t,chn,Y,l);
    [~,sns] =  check_MIMO_SNS(one_V,Pmax);
end
% catch
%     a = a;
% end
fin_eta = ub;
