%% 本程序主要实现TM-CFAR恒虚警率检测算法的函数形式
function result = func_cfar_tm(x,alpha,NSlide,Pro_cell,rate1,rate2)
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
    persistent K1;
    persistent K2;
    
    if isempty(left)
        left = 1 + Half_Pro_cell + Half_Slide; %设置左边界
        right = length(x) - Half_Pro_cell - Half_Slide; %设置右边界
        Half_Slide = NSlide / 2; %半滑动窗
        Half_Pro_cell = Pro_cell / 2; %一侧保护单元长度
        len = length(x); %杂波单元
        K1 = round(rate1 * NSlide); %有序序号
        K2 = round(rate2 * NSlide); %有序序号
    end
    
    T = zeros(1,len); %CMLD检测阈值
    target = java.util.LinkedList; %利用Java链表来实现目标的存储

    for i = 1:left - 1 %左边界
        cell_right = x(1,i + Half_Pro_cell + 1 : i + Half_Slide * 2 + Half_Pro_cell); 
        cell_right = sort(cell_right);
        cell_right(1:K1) = [];
        cell_right(end-K2+1:end) = [];
        T(1,i) = mean(cell_right) * alpha; %TM CFAR的门限
        if T(1,i) < x(i)
            target.add(i); %加入目标
        end
    end
    
    for i = left:right %中间区域
        cell_left = x(1,i - Half_Pro_cell - Half_Slide : i - Half_Pro_cell - 1);
        cell_right = x(1,i + Half_Pro_cell + 1 : i + Half_Pro_cell + Half_Slide);
        cell = [cell_left,cell_right];
        cell = sort(cell);
        cell(1:K1) = [];
        cell(end+1-K2:end) = [];
        T(1,i) = mean(cell) * alpha; %TM CFAR的门限
        if T(1,i) < x(i)
            target.add(i); %加入目标
        end
    end
    
    for i = right + 1 : len
        cell_left = x(1,i - Half_Pro_cell - Half_Slide * 2 : i - Half_Pro_cell - 1); 
        cell_left = sort(cell_left);
        cell_left(1:K1) = [];
        cell_left(end-K2+1:end) = [];
        T(1,i) = mean(cell_left) * alpha; %TM CFAR的门限
        if T(1,i) < x(i) 
            target.add(i);
        end
    end
    
    result = {'CFAR_TM',T,target}; %采用字典类型
end
    
    