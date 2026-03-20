function v = MMSE_PAM(S, P, snr)

Max = 100; 
v = zeros(size(snr));
for i=1:length(snr)
    if snr(i)<1000 
        v(i)  = 1-1/sqrt(2*pi) * integral(@(x) Inte_fun(x, S, P, snr(i)),-Max, Max);    %1/2/sqrt(pi)
    end
end

%%
function f_y = Inte_fun(R, S, P, snr)
    
  R_mat = ones(length(S),1)*R;
  S_mat = S.'*ones(1,length(R));
  P_mat = P.'*ones(1,length(R));
  
  tem = -abs(R_mat - sqrt(snr)*S_mat).^2/2; 
  tem_max = max(tem,[],1);
  tem_max2 = ones(length(S),1) * tem_max;
  P_exp = P_mat.* exp(tem-tem_max2);
  f_y = exp(tem_max) .* (abs(sum(S_mat.*P_exp)).^2 ./ sum(P_exp));
 
end
         
end