%% ICOS-CFAR恒虚警率检测算法的函数实现
function result = func_cfar_icos(x,Pfa,NSlide,Pro_cell,rate)
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
        alpha = os_threhold(Pfa,length(cell_right),rate); %初始alpha
        for time = 1:30 %IC迭代更新次数
            cell_right = sort(cell_right);
            t = cell_right(ceil(rate*length(cell_right))) * alpha; %阈值系数提取
            cell_right = cell_right(cell_right <= t);
            if os_threhold(Pfa,length(cell_right),rate) == alpha
                break; %如果alpha 更新仍相等,说明迭代收敛
            else
                alpha = os_threhold(Pfa,length(cell_right),rate); %否则更新alpha
            end
        end
        T(1,i) = cell_right(ceil(rate*length(cell_right))) * alpha;
        if T(1,i) < x(i)
            target.add(i); %加入目标
        end
    end
    
    for i = left:right %中间区域
        cell_left = x(1,i - Half_Slide - Half_Pro_cell:i - Half_Pro_cell - 1);
        cell_right = x(1,i + Half_Pro_cell + 1 : i + Half_Pro_cell + Half_Slide);
        cell = [cell_left,cell_right]; %合并两侧窗
        alpha = os_threhold(Pfa,length(cell),rate); %初始alpha
        for time = 1:30 %IC迭代更新次数
            cell = sort(cell);
            t = cell(ceil(rate*length(cell))) * alpha; %阈值系数提取
            cell = cell(cell <= t);
            if os_threhold(Pfa,length(cell),rate) == alpha
                break; %如果alpha 更新仍相等,说明迭代收敛
            else
                alpha = os_threhold(Pfa,length(cell),rate); %否则更新alpha
            end
        end
        T(1,i) = cell(ceil(rate*length(cell))) * alpha; %求解门限
        if T(1,i) < x(i)
            target.add(i); %加入目标
        end
    end
    
    for i = right + 1:len %右边界
        cell_left = x(1,i - Half_Pro_cell - Half_Slide * 2 : i-Half_Pro_cell-1); 
        alpha = os_threhold(Pfa,length(cell_left),rate); %初始alpha
        for time = 1:30 %IC迭代更新次数
            cell_left = sort(cell_left);
            t = cell_left(ceil(rate*length(cell_left))) * alpha; %阈值系数提取
            cell_left = cell_left(cell_left <= t);
            if os_threhold(Pfa,length(cell_left),rate) == alpha
                break; %如果alpha 更新仍相等,说明迭代收敛
            else
                alpha = os_threhold(Pfa,length(cell_left),rate); %否则更新alpha
            end
        end
        T(1,i) = cell_left(ceil(rate*length(cell_left))) * alpha;
        if T(1,i) < x(i)
            target.add(i);
        end
    end
    
    result = {'CFAR_ICOS',T,target}; %采用字典类型
end