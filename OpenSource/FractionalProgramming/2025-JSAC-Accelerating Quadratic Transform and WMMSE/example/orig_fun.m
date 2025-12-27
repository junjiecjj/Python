function [output] = orig_fun( N, d, A, B, X )
%ORIG_FUN 此处显示有关此函数的摘要
%   此处显示详细说明

output = 0;

for n = 1:N
    D = eye(d);
    for m = 1:N
        D = D + B(:,:,m,n)*X(:,:,m)*X(:,:,m)'*B(:,:,m,n)';
    end
    
    numerator = A(:,:,n)*X(:,:,n)*X(:,:,n)'*A(:,:,n)';
    
    output = output + trace(D\numerator);
end


end