function [temp,H_norm]=Channel_norm(H)
    N=size(H,1);
    [~,dia,~]=svd(H);
    dia=diag(dia);
    temp=sum(dia.^2)/N;
    H_norm=H/sqrt(temp);
end
