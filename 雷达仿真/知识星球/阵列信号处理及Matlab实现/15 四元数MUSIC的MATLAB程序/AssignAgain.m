%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**程序名字：重新安排四元数同构矩阵的元素位置
%**作者：    汪飞
%**日期：    2006-4-17
%**修改人：
%**日期：
%**描述：    this function just is used to assign isomorphic quaternion 
%            matrix again, closed four element of the orgin isomorphic 
%            matrix is made of the element of quaternion. it can be seen
%            as [x1,x2;-x2',x1]; The output is arranged isomorphic matrix.
%            Output = [A1,A2;-A2',A1] where origin quaternion matrix 
%            is A1+A2j
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Output = AssignAgain(S_C_Q)

[a,b] = size(S_C_Q);

% 将四元数的同构复数中对应位置的元素摆放在一起构成对应的同构复矩阵
A1 = zeros(a/2, b/2);
A2 = zeros(a/2, b/2);
A3 = zeros(a/2, b/2);
A4 = zeros(a/2, b/2);
A = zeros(a,b);

% 将每个四元数的同构复数中左上角的元素放到A1中
m = 0;
for p = 1:2:a-1
    m = m + 1;
    n = 0;
    for q = 1:2:b-1
        n = n + 1;
        A1(m,n) = S_C_Q(p,q);
    end
end

% 将每个四元数的同构复数中右上角的元素放到A2中
m = 0;
for p = 1:2:a-1
    m = m + 1;
    n = 0;
    for q = 2:2:b
        n = n + 1;
        A2(m,n) = S_C_Q(p,q);
    end
end

% 将每个四元数的同构复数中左下角的元素放到A3中
m = 0;
for p = 2:2:a
    m = m + 1;
    n = 0;
    for q = 1:2:b-1
        n = n + 1;
        A3(m,n) = S_C_Q(p,q);
    end
end

% 将每个四元数的同构复数中右下角的元素放到A4中
m = 0;
for p = 2:2:a
    m = m + 1;
    n = 0;
    for q = 2:2:b
        n = n + 1;
        A4(m,n) = S_C_Q(p,q);
    end
end

A = [A1, A2;A3,A4];
Output = A;