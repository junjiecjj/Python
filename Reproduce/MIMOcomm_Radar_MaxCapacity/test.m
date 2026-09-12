clear all;
clc;

cvx_clear;
cvx_solver sedumi

n = 3;

cvx_begin sdp
    variable X(n,n) hermitian semidefinite

    maximize(det_rootn(eye(n)+X))

    subject to
        trace(X) <= 1;
cvx_end

X
cvx_status
cvx_optval