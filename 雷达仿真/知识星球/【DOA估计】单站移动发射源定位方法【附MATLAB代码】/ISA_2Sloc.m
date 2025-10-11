function [ pos, vel ] = ISA_2Sloc( senPos, fc, ts, doam, fdm, Q )
% Y. Sun and Q. Wan, "Position Determination for Moving Transmitter Using
% Single Station," IEEE Access, vol. 6, no. 1, pp. 61103-61116, Oct. 2018.

%   Detailed explanation goes here

K = length(doam);
wc = 3e8/fc;

ukT = [sin(doam),-cos(doam)];
pk_barT = [cos(doam),sin(doam)];

h1 = ukT*senPos;
h2 = wc*fdm;
h = [h1;h2];

G1 = [ukT,ukT.*ts'];
G2 = [zeros(K,2),-pk_barT];
G = [G1;G2];

W = eye(2*K);

A = [-G,h];
[V1,~,~] = svd(A'*A);
psi = V1(1:end-1,end)/V1(end,end);

for ir = 1:2
    % update weighting matrix
    for k = 1:K
        b(k) = pk_barT(k,:)*(senPos-psi(1:2)-ts(k)*psi(3:4));
    end
    B1 = diag(b);
    B2 = -diag(ukT*psi(3:4));
    B3 = wc*eye(K);
    B = [B1,zeros(K);B2,B3];
    
    [V2,~,~] = svd(A'/(B*Q*B')*A);
    psi = V2(1:end-1,end)/V2(end,end);
end

pos = psi(1:2);
vel = psi(3:4);


end

