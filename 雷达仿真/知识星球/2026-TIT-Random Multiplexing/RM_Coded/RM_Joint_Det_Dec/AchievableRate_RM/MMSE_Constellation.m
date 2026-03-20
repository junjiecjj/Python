function v = MMSE_Constellation(S, P, snr)

Max = 24.5; 
v = zeros(size(snr));
for i=1:length(snr)
    if snr(i)<500 
        v(i)  = 1-1/pi * integral2(@(x,y) Inte_fun(x, y, S, P, snr(i)),-Max, Max,-Max, Max);    
    end
end

%%
function f_y = Inte_fun(R, I, S, P, snr)
    
 R_M = R(:,:,ones(1,length(S)));
 I_M = I(:,:,ones(1,length(S)));
 [m, n] = size(R);
 tem = S.'* ones(1,m);
 S_tem = tem(:,:,ones(1,n));
 S_M = permute(S_tem, [2,3,1]);
 
 tem_p = P.' * ones(1,m);
 P_tem = tem_p(:,:,ones(1,n));
 P_M = permute(P_tem, [2,3,1]);
  
 tem = -abs(R_M + I_M*1i - sqrt(snr)*S_M).^2; 
 tem2 = max(tem,[],3);
 tem2_M = tem2(:,:,ones(1,length(S)));
 P_exp = P_M.*exp(tem - tem2_M);
 f_y = exp(tem2) .* abs(sum(S_M.*P_exp,3)).^2 ./ sum(P_exp,3);
        
end

end