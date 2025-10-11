%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**�������֣����°�����Ԫ��ͬ�������Ԫ��λ��
%**���ߣ�    ����
%**���ڣ�    2006-4-17
%**�޸��ˣ�
%**���ڣ�
%**������    this function just is used to assign isomorphic quaternion 
%            matrix again, closed four element of the orgin isomorphic 
%            matrix is made of the element of quaternion. it can be seen
%            as [x1,x2;-x2',x1]; The output is arranged isomorphic matrix.
%            Output = [A1,A2;-A2',A1] where origin quaternion matrix 
%            is A1+A2j
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Output = AssignAgain(S_C_Q)

[a,b] = size(S_C_Q);

% ����Ԫ����ͬ�������ж�Ӧλ�õ�Ԫ�ذڷ���һ�𹹳ɶ�Ӧ��ͬ��������
A1 = zeros(a/2, b/2);
A2 = zeros(a/2, b/2);
A3 = zeros(a/2, b/2);
A4 = zeros(a/2, b/2);
A = zeros(a,b);

% ��ÿ����Ԫ����ͬ�����������Ͻǵ�Ԫ�طŵ�A1��
m = 0;
for p = 1:2:a-1
    m = m + 1;
    n = 0;
    for q = 1:2:b-1
        n = n + 1;
        A1(m,n) = S_C_Q(p,q);
    end
end

% ��ÿ����Ԫ����ͬ�����������Ͻǵ�Ԫ�طŵ�A2��
m = 0;
for p = 1:2:a-1
    m = m + 1;
    n = 0;
    for q = 2:2:b
        n = n + 1;
        A2(m,n) = S_C_Q(p,q);
    end
end

% ��ÿ����Ԫ����ͬ�����������½ǵ�Ԫ�طŵ�A3��
m = 0;
for p = 2:2:a
    m = m + 1;
    n = 0;
    for q = 1:2:b-1
        n = n + 1;
        A3(m,n) = S_C_Q(p,q);
    end
end

% ��ÿ����Ԫ����ͬ�����������½ǵ�Ԫ�طŵ�A4��
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