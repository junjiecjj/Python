function [dx,w]=w_arr_cal(px)
    C=px-px.';
    c=vec(C);
    dx=sort(unique(c));
    w=zeros(length(dx),1);
    for i=1:length(dx)
        p_i=(c==dx(i));
        w(i)=sum(p_i);
    end
end