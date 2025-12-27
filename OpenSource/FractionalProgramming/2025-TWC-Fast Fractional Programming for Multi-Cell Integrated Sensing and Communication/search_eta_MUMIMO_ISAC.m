function [eta] = search_eta_MUMIMO_ISAC(L_Com,DD,Pmax,eps)
% L_Com (Nt,K,Q,L);
[~,~,~,L] = size(L_Com); flags = zeros(L,1); eta = zeros(L,1);
for l = 1:L
   W = update_W_MUMIMO_ISAC(l,L_Com,DD,eta(l));
   [flags(l),P_w] =  check_feasibility_MUMIMO_ISAC(W,Pmax);
   if flags(l) == 0 % infeasible 
      eta(l) = bisection_eta_MUMIMO_ISAC(l,L_Com,DD,eps,Pmax);
   end
end
end