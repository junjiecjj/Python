function [eta] = search_eta(w,t,chn,Y,Pmax)
% t: auxiliary variable SINR (NumUser NumCell)
% Y: auxiliary variable (Rx NumUser NumCell)
% L: NumCell
% Q: NumUser in each cell
% chn: channel (Rx Tx 1 (NumUser x NumCell) NumCell)
% w: weight of each rate (NumUser NumCell)
% V: precoding vectors (Tx NumUser NumCell)
[~,~,L] = size(Y); 
eta =zeros(L,1);
V = update_V(eta,w,t,chn,Y);
[flags,sns] = check_MIMO_SNS(V,Pmax);
eps = 1e-6;
for l = 1:L
    if flags(l) == 0 % 违反能量约束
        eta(l) = bisection_eta(l,eps,chn,w,t,Y,Pmax);
    end
end
end