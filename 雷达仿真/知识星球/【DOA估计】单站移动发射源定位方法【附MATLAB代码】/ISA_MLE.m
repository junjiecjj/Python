function [ pos, vel ] = ISA_MLE( senPos, fc, ts, doam, fdm, Q, psi )
% Y. Sun and Q. Wan, "Position Determination for Moving Transmitter Using
% Single Station," IEEE Access, vol. 6, no. 1, pp. 61103-61116, Oct. 2018.

%   Detailed explanation goes here
warning off;

K = length(doam);
wc = 3e8/fc;
gamma = [doam;fdm];

Iter = 200;
for it = 1:Iter
    posk = psi(1:2) + psi(3:4).*ts;
    doak = atan2(posk(2,:)-senPos(2),posk(1,:)-senPos(1))';
    fdk = -(psi(3:4)'*(posk-senPos)./(sqrt(sum((posk-senPos).^2,1)))/wc)';
    gammak = [doak;fdk];

    r_k = sqrt(sum((posk-senPos).^2,1));
    for k = 1:K
        U(k,1) = -(posk(2,k)-senPos(2))/r_k(k)^2;
        U(k,2) = (posk(1,k)-senPos(1))/r_k(k)^2;
        U(k,3:4) = ts(k)*U(k,1:2);

        u_k = [sin(doak(k));-cos(doak(k))];
        U(k+K,:) = -1/wc*( [0,0,cos(doak(k)),sin(doak(k))] - u_k'*psi(3:4)*U(k,:) );
    end

    psi_new = psi + (U'/Q*U)\U'/Q*(gamma-gammak);
    if norm(psi_new-psi,2) < 1e-9
        psi = psi_new;
        break;
    end
    psi = psi_new;
end

pos = psi(1:2);
vel = psi(3:4);

end

