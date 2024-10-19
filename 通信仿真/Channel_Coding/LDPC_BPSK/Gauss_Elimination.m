
function [H,exchangej] = Gauss_Elimination(H)

[m,n] = size(H);
exchangej = 1:n;
for i=1:m % 逐一检查H的左边待单位化的矩阵的主对角线元素，若为0，则优先行交换，再列交换，使得该元素非零
    if H(i,i)==0 % 若H(i,i)==0
        j = i + find(H(i+1:m,i), 1 );% 优先行交换，寻找该列中的第一个非零元，记录该非零元的行（H(i+1:m,i)只对该列H(i,i)后面的元素寻找，因为在i上面的行已经交换好不可再改变）
        H([j i],:) = H([i,j],:);% 将H的该行与H的第i行交换
        if isempty(j)
            j = i + find(H(i,i+1:n), 1 );%寻找该行中第一个非零元，记录该非零元的列 ,如果也为空,那就go die吧
            if isempty(j)
                error('Matrix is not full rank, systematic matrix can not be made');
            end
            H(:,[j i]) = H(:,[i,j]);  %交换列
            exchangej([i j]) = exchangej([j i]);%置换后的下标
        end
        
        for k=i+1:m % 将H(i,i)元素以下的非零元采用行叠加变零
            if H(k,i)==1 % 有，则将H的第i行叠加到该行
                H(k,:)=H(i,:)+H(k,:);
                H(k,:)=mod(H(k,:),2); %%%%进行叠加处！！！！！
            end % 无,则检查H的下一行
        end
    else % 若H(i,i)==1，不需要交换行，只需要检查H(i,i)元素以下的第i+1行到m行是否还有非零元
        for k=i+1:m %将H(i,i)元素以下的非零元采用行叠加变零
            if H(k,i)==1 % 有，则将H的第i行叠加到该行
                H(k,:)=H(i,:)+H(k,:);
                H(k,:)=mod(H(k,:),2);  %%%%进行叠加处！！！！！
            end
        end
    end  % 无,则检查H的下一行
end


for i=m:-1:1 % 自第m行向第j行叠加,j=m-1:-1:1 ，将H左边待单位化的矩阵的主对角线上面的元素通过行叠加变零
    for k=i-1:-1:1
        if H(k,i)==1
            H(k,:)=H(i,:)+H(k,:);
            H(k,:)=mod(H(k,:),2);   %%%%进行叠加处！！！！！
        end
    end % 循环之后得到左半为单位阵的矩阵H = [I|P]
end


