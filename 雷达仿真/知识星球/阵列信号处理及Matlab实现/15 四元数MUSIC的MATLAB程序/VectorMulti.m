%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**�������֣���Ԫ����ʸ����
%**���ߣ�    ����
%**���ڣ�    2006-6-10
%**�޸��ˣ�
%**���ڣ�      
%**������    ����Q_MUSIC����
%**            
%**         �˴�ColΪ��������RowΪ������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output = VectorMulti(Col, Row);

% ����ÿһ�ж���ͬ�ķ���,��a X a*b
[a,b] = size(Col);
ColToSquare = Col;
for i = 1:a-1
    ColToSquare = [ColToSquare,Col];
end

% ��Row����ÿһ��
c = zeros(a,a*b);
m = 0;
for p = 1:a
    m = m + 1;
    n = 0;
    for q = 1:a
        n = n + 1;
        d = ColToSquare(m,(n-1)*4+1:n*4);
        e = Row(1,(n-1)*4+1:n*4);
        c(m,(n-1)*4+1:n*4) = hpc(d,e);
    end
end

output = c;
        
        




    
    