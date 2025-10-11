function [ pos, vel ] = ISA_2SBRloc( senPos, fc, ts, doam, fdm, Q )
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
[V1,~,~] = svd(A'*W*A);
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
    
    W = eye(2*K)/(B*Q*B');
    
    P1 = [pk_barT,pk_barT.*ts'];
    P2 = [zeros(K,2),ukT];
    P3 = diag(pk_barT*senPos);
    
    Qa = Q(1:K,1:K);
    W1 = diag(diag(W(1:K,1:K)));
    W2 = diag(diag(W(1:K,(1:K)+K)));
    W3 = diag(diag(W((1:K)+K,1:K)));
    W4 = diag(diag(W((1:K)+K,(1:K)+K)));
    
    O1 = P1'*(Qa.*W1)*P1 + P1'*(Qa.*W2)*P2 + P2'*(Qa.*W3)*P1 + P2'*(Qa.*W4)*P2;
    O2 = (P1'*(Qa.*(W1*P3)) + P2'*(Qa.*(W3*P3)))*ones(K,1);
    O3 = trace(P3'*W1*P3*Q(1:K,1:K)) + wc*2*trace(W4*Q((1:K)+K,(1:K)+K));
    O = [O1,-O2;-O2',O3];
       
    
    [V2,~] = eig(A'*W*A,O);
    sol = zeros(5);
    for i = 1:5    
        sol(:,i) = V2(1:end,i)/V2(end,i);
        J(i) = sol(:,i)'*A'*W*A*sol(:,i);
    end
    [~,ind] = min(J);
    psi = sol(1:end-1,ind);
end

pos = psi(1:2);
vel = psi(3:4);


end

