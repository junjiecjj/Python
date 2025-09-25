function [eta1,eta2] = find_eta(A,alpha,ys,w1,y1,H11,w2,y2,H22,D1,D2,P,M)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
eta1 = 0; eta2 = 0;
v1 = (eta1*eye(M)+D1)\(alpha*A'*ys+w1*(H11')*y1);
v2 = (eta2*eye(M)+D2)\(w2*(H22')*y2);
%% search eta1
if norm(v1)^2>P
    % find the proper iterval to perform bisection
    eta1 = 2;
    v1 = (eta1*eye(M)+D1)\(alpha*A'*ys+w1*(H11')*y1);
    while norm(v1)^2>P
        eta1 = eta1^2;
        v1 = (eta1*eye(M)+D1)\(alpha*A'*ys+w1*(H11')*y1);
    end
    if eta1 == 2
    l = 0;
    else
        l = sqrt(eta1);
    end
    u = eta1;
    v1 = (u*eye(M)+D1)\(alpha*A'*ys+w1*(H11')*y1);
    while (P-norm(v1)^2)/P>1e-10
        mid  = (l+u)/2;
        v1 = (mid*eye(M)+D1)\(alpha*A'*ys+w1*(H11')*y1);
        if norm(v1)^2>P
            l = mid;
        else
            u = mid;
        end
        v1 = (u*eye(M)+D1)\(alpha*A'*ys+w1*(H11')*y1);
    end
    eta1 = u;
end
%% search eta2
if norm(v2)^2>P
    % find the proper iterval to perform bisection
    eta2 = 2;
    v2 = (eta2*eye(M)+D2)\(w2*(H22')*y2);
    while norm(v2)^2>P
        eta2 = eta2^2;
        v2 = (eta2*eye(M)+D2)\(w2*(H22')*y2);
    end
    if eta2 == 2
    l = 0;
    else
        l = sqrt(eta2);
    end
   u = eta2;
    v2 = (eta2*eye(M)+D2)\(w2*(H22')*y2);
    while (P-norm(v2)^2)/P>1e-3
        mid  = (l+u)/2;
        v2 = (mid*eye(M)+D2)\(w2*(H22')*y2);
        if norm(v2)^2>P
            l = mid;
        else
            u = mid;
        end
        v2 = (u*eye(M)+D2)\(w2*(H22')*y2);
    end
    eta2 = u;
end
end