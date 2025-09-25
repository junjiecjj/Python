function [obj,FI,SINR1,SINR2] = Compute_obj_JSAC(v1,v2,H11,H22,sigma1,sigma2,H12,H21,Nr,G,sigma_s,alpha,A,w1,w2)
[N,~] = size(H12);
Q1 = sigma1*eye(N)+(H12*v2)*(H12*v2)';
Q1 = (Q1+Q1')/2;
Q2 = sigma2*eye(N)+(H21*v1)*(H21*v1)';
Q2 = (Q2+Q2')/2;

SINR1 = v1'*H11'*inv(Q1)*H11*v1;
SINR2 = v2'*H22'*inv(Q2)*H22*v2;

Q3 = sigma_s*eye(Nr)+(G*v2)*(G*v2)';
Q3 = (Q3+Q3')/2;

FI = alpha*v1'*A'*inv(Q3)*A*v1;
SINR1 = real(SINR1); SINR2 = real(SINR2); FI = real(FI);
obj = FI+w1*SINR1+w2*SINR2;
end