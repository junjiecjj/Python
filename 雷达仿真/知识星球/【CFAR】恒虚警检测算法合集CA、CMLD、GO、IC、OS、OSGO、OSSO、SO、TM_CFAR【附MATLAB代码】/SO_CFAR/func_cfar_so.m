%% SO-CFAR恒虚警率检测算法的函数实现
function result = func_cfar_so(x,alpha,NSlide,Pro_cell)
    %x:原始杂波数据
    %alpha：标称因子
    %Nslide：滑动窗大小
    %Pro_cell:保护单元大小
    
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
%     target = java.util.LinkedList; %利用Java链表来实现目标的存储
    target = [];
    
    for i = 1:left - 1 %左边界
        cell_right = x(1,i + Half_Pro_cell + 1 : i + Half_Slide * 2 + Half_Pro_cell); 
        T(1,i) = mean(cell_right) * alpha;
        if T(1,i) < x(i)
            target = [target, i]; %加入目标
        end
    end
    
    for i = left:right %中间区域
        cell_left = x(1,i - Half_Slide - Half_Pro_cell:i - Half_Pro_cell - 1);
        cell_right = x(1,i + Half_Pro_cell + 1 : i + Half_Pro_cell + Half_Slide);
        T(1,i) = min(mean(cell_left),mean(cell_right)) * alpha; %求解门限
        if T(1,i) < x(i)
            target = [target, i]; %加入目标
        end
    end
    
    for i = right + 1:len %右边界
        cell_left = x(1,i - Half_Pro_cell - Half_Slide * 2 : i-Half_Pro_cell-1); 
        T(1,i) = mean(cell_left) * alpha;
        if T(1,i) < x(i)
            target = [target, i]; %加入目标
        end
    end
    
    result = {'CFAR_SO',T,target}; %采用字典类型
end