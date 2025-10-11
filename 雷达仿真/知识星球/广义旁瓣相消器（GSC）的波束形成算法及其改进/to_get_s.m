function s=to_get_s(w,N,p)
    s=zeros(p,N);
    for i=1:p
        s(i,1:N)=exp(j*w(i).*(1:N)); % 复指数信号  假设信道增益为 1
    end
end
