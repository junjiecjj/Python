%% ICCA-CFAR恒虚警率检测算法的函数实现
function result = func_cfar_icca(x,Pfa,NSlide,Pro_cell)
    %x:原始杂波数据
    %alpha：标称因子
    %Nslide：滑动窗大小
    %Pro_cell:保护单元大小
    %rate:有序比例，OS中选择序号如k=3/4N时rate=0.75
    
    persistent left; 
    persistent right;
    persistent Half_Slide;
    persistent Half_Pro_cell;
    persistent len;
    
    if isempty(left)
        left = 1 + Half_Pro_cell + Half_Slide; %设置左边界
        right = length(x) - Half_Pro_cell - Half_Slide; %设置右边界
        Half_Slide = NSlide / 2; %半滑动窗
        Half_Pro_cell = Pro_cell / 2; %一侧保护单元长度
        len = length(x); %杂波单元
    end
    T = zeros(1,len); %检测阈值
    target = java.util.LinkedList; %利用Java链表来实现目标的存储
    
    for i = 1:left - 1 %左边界
        cell_right = x(1,i + Half_Pro_cell + 1 : i + Half_Slide * 2 + Half_Pro_cell); 
        alpha = 1 - Pfa^(1 / (NSlide-1)); %初始alpha0
        for time = 1:30 %IC迭代更新次数
            t = sum(cell_right) * alpha;%阈值系数提取
            cell_right = cell_right(cell_right <= t);
            if 1 - Pfa^(1 / (length(cell_right)-1)) == alpha
                break; %如果alpha 更新仍相等,说明迭代收敛
            else
                alpha = 1 - Pfa^(1 / (length(cell_right)-1)); %否则更新alpha
            end
        end
        T(1,i) = sum(cell_right) * alpha;
        if T(1,i) < x(i)
            target.add(i); %加入目标
        end
    end
    
    for i = left:right %中间区域
        cell_left = x(1,i - Half_Slide - Half_Pro_cell:i - Half_Pro_cell - 1);
        cell_right = x(1,i + Half_Pro_cell + 1 : i + Half_Pro_cell + Half_Slide);
        cell = [cell_left,cell_right]; %合并两侧窗
        alpha = 1 - Pfa^(1 / (NSlide-1)); %初始alpha0
        for time = 1:30 %IC迭代更新次数
            t = sum(cell) * alpha;%阈值系数提取
            cell = cell(cell <= t);
            if 1 - Pfa^(1 / (length(cell)-1)) == alpha
                break; %如果alpha 更新仍相等,说明迭代收敛
            else
                alpha = 1 - Pfa^(1 / (length(cell)-1)); %否则更新alpha
            end
        end
        T(1,i) = sum(cell) * alpha; %求解门限
        if T(1,i) < x(i)
            target.add(i); %加入目标
        end
    end
    
    for i = right + 1:len %右边界
        cell_left = x(1,i - Half_Pro_cell - Half_Slide * 2 : i-Half_Pro_cell-1); 
        alpha = 1 - Pfa^(1 / (NSlide-1)); %初始alpha0
        for time = 1:30 %IC迭代更新次数
            t = sum(cell_left) * alpha;%阈值系数提取
            cell_left = cell_left(cell_left <= t);
            if 1 - Pfa^(1 / (length(cell_left)-1)) == alpha
                break; %如果alpha 更新仍相等,说明迭代收敛
            else
                alpha = 1 - Pfa^(1 / (length(cell_left)-1)); %否则更新alpha
            end
        end
        T(1,i) = sum(cell_left) * alpha;
        if T(1,i) < x(i)
            target.add(i);
        end
    end
    
    result = {'CFAR_ICCA',T,target}; %采用字典类型
end