function z=v_arr_cal(Y,px)
    M=length(px);
    C=px-px.';
    y=vec(Y);
    c=vec(C);
    dx=sort(unique(c));
    z0=zeros(length(dx),1);
    for i=1:length(dx)
        p_i=(c==dx(i));
        z0(i)=sum(y(p_i))/sum(p_i);
    end
    z=z0;
end