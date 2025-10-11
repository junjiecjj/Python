%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**程序名字：四元数的矢量积
%**作者：    汪飞
%**日期：    2006-6-10
%**修改人：
%**日期：      
%**描述：    仿真Q_MUSIC方法
%**            
%**         此处Col为列向量，Row为横向量
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output = VectorMulti(Col, Row);

% 构造每一列都相同的方阵,即a X a*b
[a,b] = size(Col);
ColToSquare = Col;
for i = 1:a-1
    ColToSquare = [ColToSquare,Col];
end

% 将Row乘上每一行
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
        
        




    
    