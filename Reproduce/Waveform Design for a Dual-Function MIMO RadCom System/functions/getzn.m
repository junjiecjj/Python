




% function zn = getzn(Z, M, P, n)
%     zn = zeros(M, 1);
%     for p = 1:P
%         zn = zn + Z(p:p+M-1, (n-1)*P + p);
%     end
%     zn = zn / P;
% end

function zn = getzn(Z, M, P, n)
    zn = zeros(1, M);
    for p = 1:P
        zn = zn + Z((n-1)*P + p, p:p+M-1);
    end
    zn = zn / P;
end
